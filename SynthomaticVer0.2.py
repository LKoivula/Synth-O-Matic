# Synth-O-Matic 0.2.0-beta -- single-file 3D Slicer console script
# Copy this ENTIRE file into Slicer's Python console and execute it.
# The exec wrapper makes the paste one statement, including all blank lines.
exec(compile(r'''
"""MRI -> N4 -> synthetic CT. Research beta; not clinically validated.

Requirements: 3D Slicer with N4; SlicerRT and pydicom for DICOM export.
The GUI, N4 and SlicerRT execution require testing in the target Slicer version.
After export, visually inspect CT/RTSTRUCT in Slicer and the intended receiver.

Input: MRI-image (Dixon InPhase image) and segmentations: Soft tissue, Bone, Whole BODY 
(from Slicer extension Totalsegmentator for example).

Output: Synthetic CT image with HU values and RT structures in DICOM format.

Edit SETTINGS below to match your VOLUME and SEGMENT names. 

Workflow: launch runs N4 and creates a preview. Update re-reads segmentation
and applies the current fields. A changed bone mask refits MIN/P50/MAX anchors.
A changed MRI or BODY requires 'Run N4 again'. Export recalculates and uses a
single snapshot of HU data, effective masks, geometry and parameters.

Conversion model: soft-tissue plateaus with linear transitions; bone mapping
exp(a+b*x+c*x*x)-1, anchored at MIN/P50/MAX MRI intensity values. N4 does not 
standardize MRI intensity across acquisitions. Unassigned BODY voxels remain -1000 HU.

Arrays use Slicer/NumPy KJI order. Geometry uses IJK-to-RAS. This beta rejects
parent transforms and sheared grids rather than silently losing geometry.
Harden/resample transforms explicitly in Slicer before starting if needed.
No input nodes are overwritten or removed. Repeated launches close the previous
window but retain its result volumes. Scene close requires relaunching.

Developer map: pure NumPy functions -> Slicer adapter -> DICOM -> dialog.
Only the final launcher touches the shared console namespace. Its private
execution namespace keeps old callbacks isolated when the file is pasted again.
"""
import datetime
import hashlib
import json
import logging
import uuid
from pathlib import Path

import numpy as np
import qt
import slicer
import vtk

VERSION = "0.2.0-beta"
LOG = logging.getLogger("SynthOMatic")
SETTINGS = {
    "input_volume": "InPhase",
    "segmentation": "Pelvis_segmentation",
    "segments": {"body": "BODY", "soft": "SOFT_TISSUE", "bone": "BONES"},
    "n4_shrink_factor": 4,
    "outside_hu": -1000, "urine_hu": 10, "muscle_hu": 45, "fat_hu": -100, #HU values for air, urine, muscle, and fat
    "urine_max": 200, "muscle_min": 250, "muscle_max": 450, "fat_min": 650, #preset MRI intensity value limits
    "bone_targets": [1350, 500, 0],
    "bone_min_hu": 0, "bone_max_hu": 3000,
}
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


# ---- Pure calculation: no MRML nodes or GUI fields --------------------------
def validate_parameters(p):
    """Raise ValueError for invalid thresholds, HU limits or nonfinite values."""
    keys = ("outside_hu", "urine_hu", "muscle_hu", "fat_hu", "urine_max",
            "muscle_min", "muscle_max", "fat_min", "bone_min_hu", "bone_max_hu")
    if not all(np.isfinite(p[k]) for k in keys):
        raise ValueError("All conversion parameters must be finite.")
    if not p["urine_max"] < p["muscle_min"] <= p["muscle_max"] < p["fat_min"]:
        raise ValueError("Required: urine max < muscle min <= muscle max < fat min.")
    hu_keys = ("outside_hu", "urine_hu", "muscle_hu", "fat_hu",
               "bone_min_hu", "bone_max_hu")
    if not all(-32768 <= p[k] <= 32767 for k in hu_keys):
        raise ValueError("HU values must fit signed 16-bit storage.")
    if not 0 <= p["bone_min_hu"] < p["bone_max_hu"]:
        raise ValueError("Required: 0 <= bone minimum HU < bone maximum HU.")


def effective_masks(body, soft, bone):
    """Return independent Boolean masks; bone has priority over soft tissue."""
    if body.shape != soft.shape or body.shape != bone.shape:
        raise ValueError("Mask dimensions do not match.")
    body, soft, bone = [np.asarray(a, dtype=bool) for a in (body, soft, bone)]
    result = {"body": body.copy(), "soft": soft & body & ~bone, "bone": bone & body}
    for name, mask in result.items():
        if not mask.any():
            raise ValueError(f"The effective {name} mask is empty.")
    return result


def bone_anchors(values):
    return np.array([np.min(values), np.percentile(values, 50), np.max(values)],
                    dtype=np.float64)


def fit_bone(anchors, targets):
    """Exact three-point fit in log(HU+1), preserving the original arithmetic.

    The quadratic exponent need not be monotone. Coefficients remain in raw
    intensity units so existing a/b/c values keep their original meaning.
    """
    x, y = np.asarray(anchors, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if x.shape != (3,) or y.shape != (3,) or not np.isfinite([x, y]).all():
        raise ValueError("Three finite MRI anchors and HU targets are required.")
    if np.any(y < 0) or np.any(np.diff(x) < 1e-10):
        raise ValueError("Bone anchors must be distinct and increasing; HU targets >= 0.")
    return np.linalg.solve(np.column_stack([np.ones(3), x, x*x]), np.log(y + 1.0))


def bone_hu(values, coefficients, p):
    """Apply the original exponential mapping and HU clipping."""
    a, b, c = coefficients
    exponent = np.clip(a + b*values + c*values*values, -50, 50)
    return np.clip(np.exp(exponent) - 1.0, p["bone_min_hu"], p["bone_max_hu"])


def bone_is_decreasing(anchors, coefficients):
    # The derivative sign is b+2*c*x, linear in x; endpoints suffice.
    a, b, c = coefficients
    return bool(np.all(b + 2*c*np.asarray(anchors)[[0, 2]] <= 0))


def calculate_sct(image, masks, p, coefficients):
    """Return int16 HU in KJI order. Never mutate inputs or touch the GUI.

    Unassigned voxels, including those inside BODY, retain outside_hu. This
    preserves the prior behaviour; gas and segmentation gaps are not inferred.
    """
    validate_parameters(p)
    if not np.isfinite(coefficients).all() or len(coefficients) != 3:
        raise ValueError("Three finite bone coefficients are required.")
    if image.ndim != 3 or any(m.shape != image.shape for m in masks.values()):
        raise ValueError("A 3D image and matching masks are required.")
    if not np.isfinite(image).all():
        raise ValueError("MRI contains NaN or infinity; correct the input first.")
    soft = image[masks["soft"]]
    hu = np.full(soft.shape, np.nan, dtype=np.float32)
    u, lo, hi, f = [p[k] for k in ("urine_max", "muscle_min", "muscle_max", "fat_min")]
    hu[soft <= u] = p["urine_hu"]
    m = (soft > u) & (soft < lo)
    hu[m] = p["urine_hu"] + ((soft[m]-u)/(lo-u))*(p["muscle_hu"]-p["urine_hu"])
    hu[(soft >= lo) & (soft <= hi)] = p["muscle_hu"]
    m = (soft > hi) & (soft < f)
    hu[m] = p["muscle_hu"] + ((soft[m]-hi)/(f-hi))*(p["fat_hu"]-p["muscle_hu"])
    hu[soft >= f] = p["fat_hu"]
    output = np.full(image.shape, p["outside_hu"], dtype=np.float32)
    output[masks["soft"]] = hu
    output[masks["bone"]] = bone_hu(image[masks["bone"]].astype(np.float64), coefficients, p)
    if not np.isfinite(output).all() or output.min() < -32768 or output.max() > 32767:
        raise ValueError("Calculated HU values are not finite or exceed int16 range.")
    return np.rint(output).astype(np.int16)


def fingerprint(array):
    """Content fingerprint catches in-place edits as well as replacement arrays."""
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


# ---- Slicer adapter --------------------------------------------------------
def unique_node(name, class_name):
    matches = [n for n in slicer.util.getNodesByClass(class_name) if n.GetName() == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {class_name} named '{name}', found {len(matches)}.")
    return matches[0]


def geometry(node):
    if node.GetParentTransformNode():
        raise ValueError(f"'{node.GetName()}' has a parent transform. Harden it first.")
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    result = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
    if node.GetImageData() is None:
        raise ValueError("Volume has no image data.")
    axes = result[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    if not np.isfinite(result).all() or np.any(spacing <= 0):
        raise ValueError("Invalid image geometry.")
    direction = axes/spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6):
        raise ValueError("Sheared image geometry is not supported. Resample it first.")
    extent = node.GetImageData().GetExtent()
    if extent[0] != 0 or extent[2] != 0 or extent[4] != 0:
        raise ValueError("Nonzero image extent origin is not supported in this beta.")
    return result


def remove_nodes(nodes):
    for node in reversed(nodes):
        if node is not None and node.GetScene() is not None:
            slicer.mrmlScene.RemoveNode(node)


def make_volume(reference, array, name, labelmap=False):
    kind = "vtkMRMLLabelMapVolumeNode" if labelmap else "vtkMRMLScalarVolumeNode"
    node = slicer.mrmlScene.AddNewNodeByClass(kind, name)
    node.CopyOrientation(reference)
    node.CreateDefaultDisplayNodes()
    slicer.util.updateVolumeFromArray(node, array)
    return node


def read_masks(segmentation, reference):
    if segmentation.GetParentTransformNode():
        raise ValueError("Segmentation has a parent transform. Harden it first.")
    seg = segmentation.GetSegmentation()
    arrays = {}
    for key, name in SETTINGS["segments"].items():
        ids = [seg.GetNthSegmentID(i) for i in range(seg.GetNumberOfSegments())
               if seg.GetNthSegment(i).GetName() == name]
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one segment named '{name}'.")
        arrays[key] = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation, ids[0], reference) > 0
    return effective_masks(arrays["body"], arrays["soft"], arrays["bone"])


def dicom_tag(volume, tag, default=""):
    """Read source metadata if its DICOM instances are still in the database."""
    uids = volume.GetAttribute("DICOM.instanceUIDs")
    if not uids or not slicer.dicomDatabase or not slicer.dicomDatabase.isOpen:
        return default
    path = slicer.dicomDatabase.fileForInstance(uids.split()[0])
    return (str(slicer.dicomDatabase.fileValue(path, tag)) or default) if path else default


class Workflow:
    """Own one input selection, N4 cache and the last successful result snapshot."""
    def __init__(self):
        self.input = unique_node(SETTINGS["input_volume"], "vtkMRMLScalarVolumeNode")
        self.segmentation = unique_node(SETTINGS["segmentation"], "vtkMRMLSegmentationNode")
        self.n4 = self.bias = self.output = None
        self.result = None
        self.source_hash = self.body_hash = None

    def current_inputs(self):
        if self.input.GetScene() is None or self.segmentation.GetScene() is None:
            raise ValueError("Input nodes were removed. Close this window and run the script again.")
        matrix = geometry(self.input)
        image = slicer.util.arrayFromVolume(self.input).astype(np.float32)
        if image.ndim != 3 or not image.size or not np.isfinite(image).all():
            raise ValueError("Input must be a nonempty finite 3D scalar MRI.")
        return image, matrix, read_masks(self.segmentation, self.input)

    def run_n4(self):
        """Commit a new cache only after CLI success and geometry validation."""
        image, matrix, masks = self.current_inputs()
        temporary, outputs = [], []
        try:
            mask = make_volume(self.input, masks["body"].astype(np.uint8), "N4_BODY_TEMP", True)
            temporary.append(mask)
            n4 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "InPhase_N4")
            outputs.append(n4)
            bias = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "InPhase_N4_BiasField")
            outputs.append(bias)
            cli = slicer.cli.runSync(slicer.modules.n4itkbiasfieldcorrection, None, {
                "inputImageName": self.input.GetID(), "maskImageName": mask.GetID(),
                "outputImageName": n4.GetID(), "outputBiasFieldName": bias.GetID(),
                "shrinkFactor": SETTINGS["n4_shrink_factor"],
            })
            temporary.append(cli)
            if cli.GetStatusString() != "Completed":
                raise RuntimeError("N4 failed: " + cli.GetErrorText())
            corrected = slicer.util.arrayFromVolume(n4).astype(np.float32)
            if corrected.shape != image.shape or not np.isfinite(corrected).all():
                raise ValueError("Invalid N4 output dimensions or values.")
            if not np.allclose(geometry(n4), matrix, rtol=0, atol=1e-5):
                raise ValueError("N4 output geometry differs from the input.")
            self.original, self.image, self.matrix = image, corrected, matrix
            self.n4, self.bias = n4, bias
            self.source_hash, self.body_hash = fingerprint(image), fingerprint(masks["body"])
            self.result = None
            outputs.clear()  # Successful output volumes remain in the scene.
            return masks
        finally:
            remove_nodes(temporary + outputs)

    def fresh_masks(self):
        image, matrix, masks = self.current_inputs()
        if (fingerprint(image) != self.source_hash or not np.array_equal(matrix, self.matrix)
                or fingerprint(masks["body"]) != self.body_hash):
            raise ValueError("MRI, geometry or BODY changed. Click 'Run N4 again' first.")
        if (self.n4.GetScene() is None
                or not np.array_equal(slicer.util.arrayFromVolume(self.n4).astype(np.float32), self.image)
                or not np.allclose(geometry(self.n4), self.matrix, rtol=0, atol=1e-5)):
            raise ValueError("N4 output was edited or removed. Run N4 again to rebuild the cache.")
        return masks

    def commit(self, masks, parameters, coefficients):
        data = calculate_sct(self.image, masks, parameters, coefficients)
        # New geometry comes from the cached input grid, not a manually edited preview.
        if self.output is None or self.output.GetScene() is None:
            self.output = make_volume(self.input, data, "sCT_preview")
        else:
            self.output.SetAndObserveTransformNodeID(None)
            self.output.CopyOrientation(self.input)
            slicer.util.updateVolumeFromArray(self.output, data)
        anchors = bone_anchors(self.image[masks["bone"]].astype(np.float64))
        missing = masks["body"] & ~(masks["soft"] | masks["bone"])
        info = {
            "version": VERSION, "created": datetime.datetime.now().isoformat(),
            "parameters": parameters, "coefficients": list(map(float, coefficients)),
            "bone_anchors": anchors.tolist(), "bone_targets": parameters["bone_targets"],
            "monotone": bone_is_decreasing(anchors, coefficients),
            "unassigned_voxels": int(missing.sum()),
            "unassigned_cm3": float(missing.sum()*abs(np.linalg.det(self.matrix[:3, :3]))/1000),
            "n4_shrink_factor": SETTINGS["n4_shrink_factor"],
            "input_sha256": self.source_hash, "n4_sha256": fingerprint(self.image),
            "mask_sha256": {k: fingerprint(v) for k, v in masks.items()},
            "ijk_to_ras": self.matrix.tolist(), "slicer_version": str(slicer.app.applicationVersion),
            "hu_sha256": fingerprint(data),
        }
        self.result = {"data": data, "masks": masks, "info": info}
        self.output.SetAttribute("SynthOMatic.Parameters", json.dumps(info))
        self.output.GetDisplayNode().SetAutoWindowLevel(False)
        return self.result


# ---- DICOM export and independent readback ---------------------------------
def verify_dicom(folder, result):
    """Read files without importing them into Slicer's DICOM database.

    Check exact CT HU values, pixel orientation/spacing/positions, unique UIDs,
    study/frame agreement and RTSTRUCT references. This is NOT a contour-mask
    roundtrip or validation in a treatment planning system.
    """
    import pydicom
    from pydicom.errors import InvalidDicomError
    datasets = []
    for path in Path(folder).rglob("*"):
        if path.is_file():
            try:
                ds = pydicom.dcmread(str(path))
            except InvalidDicomError:
                continue
            datasets.append(ds)
    ct = [ds for ds in datasets if getattr(ds, "Modality", "") == "CT"]
    rt = [ds for ds in datasets if getattr(ds, "Modality", "") == "RTSTRUCT"]
    data, info = result["data"], result["info"]
    if len(ct) != data.shape[0] or len(rt) != 1:
        raise ValueError("Export does not contain the expected CT slices and one RTSTRUCT.")
    uids = [str(ds.SOPInstanceUID) for ds in datasets]
    if len(uids) != len(set(uids)):
        raise ValueError("Duplicate DICOM SOP Instance UIDs.")
    if len({str(ds.StudyInstanceUID) for ds in ct + rt}) != 1:
        raise ValueError("CT and RTSTRUCT StudyInstanceUID mismatch.")
    if len({str(ds.SeriesInstanceUID) for ds in ct}) != 1:
        raise ValueError("CT slices do not belong to one series.")
    frames = {str(ds.FrameOfReferenceUID) for ds in ct}
    if len(frames) != 1:
        raise ValueError("CT FrameOfReferenceUID mismatch.")
    frame = next(iter(frames))
    matrix = np.diag([-1., -1., 1., 1.]) @ np.array(info["ijk_to_ras"])
    axes = matrix[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    orientation = np.r_[axes[:, 0]/spacing[0], axes[:, 1]/spacing[1]]
    positions = {}
    for ds in ct:
        if not np.allclose(ds.ImageOrientationPatient, orientation, rtol=0, atol=1e-5):
            raise ValueError("Export changed CT orientation; direct pixel verification failed.")
        if not np.allclose(ds.PixelSpacing, spacing[[1, 0]], rtol=0, atol=1e-4):
            raise ValueError("CT pixel spacing mismatch.")
        position = np.asarray(ds.ImagePositionPatient, dtype=float)
        ijk = np.linalg.solve(axes, position-matrix[:3, 3])
        k = int(round(ijk[2]))
        if not 0 <= k < data.shape[0] or not np.allclose(ijk, [0, 0, k], atol=1e-3):
            raise ValueError("CT slice position mismatch.")
        if k in positions:
            raise ValueError("Duplicate CT slice position.")
        actual = ds.pixel_array.astype(np.float64)*float(ds.get("RescaleSlope", 1))
        actual += float(ds.get("RescaleIntercept", 0))
        if not np.array_equal(actual, data[k]):
            raise ValueError(f"Exported HU pixels differ from the sCT snapshot at slice {k}.")
        positions[k] = str(ds.SOPInstanceUID)
    rt = rt[0]
    refs = list(getattr(rt, "ReferencedFrameOfReferenceSequence", []))
    if not refs or any(str(r.FrameOfReferenceUID) != frame for r in refs):
        raise ValueError("RTSTRUCT reference frame does not match CT.")
    roi = list(getattr(rt, "StructureSetROISequence", []))
    if {str(r.ROIName) for r in roi} != set(SETTINGS["segments"].values()):
        raise ValueError("Exported ROI names do not match the effective masks.")
    if any(str(r.ReferencedFrameOfReferenceUID) != frame for r in roi):
        raise ValueError("ROI frame does not match CT.")
    series_uid = str(ct[0].SeriesInstanceUID)
    series_refs = [series for ref in refs
                   for study in getattr(ref, "RTReferencedStudySequence", [])
                   for series in getattr(study, "RTReferencedSeriesSequence", [])]
    if not series_refs or any(str(s.SeriesInstanceUID) != series_uid for s in series_refs):
        raise ValueError("RTSTRUCT referenced series does not match CT.")
    ct_uids = set(positions.values())
    referenced = set()
    for element in rt.iterall():
        if element.keyword == "ContourImageSequence":
            for item in element.value:
                uid = str(item.ReferencedSOPInstanceUID)
                if uid not in ct_uids:
                    raise ValueError("RTSTRUCT references an image outside the exported CT.")
                referenced.add(uid)
    if not referenced:
        raise ValueError("RTSTRUCT contains no CT image references.")
    contour_rois = {int(r.ReferencedROINumber) for r in getattr(rt, "ROIContourSequence", [])
                    if len(getattr(r, "ContourSequence", [])) > 0}
    if contour_rois != {int(r.ROINumber) for r in roi}:
        raise ValueError("One or more exported ROIs have no contours.")
    return {"ct_slices": len(ct), "exact_hu_match": True, "geometry_match": True,
            "rtstruct_references_match": True,
            "contour_mask_roundtrip_performed": False}


def finalize_metadata(folder, metadata):
    """Supply descriptive tags not consistently forwarded by SlicerRT versions.

    Never alter pixel encoding, geometry, or UIDs. The synthetic CT is derived
    data. UTF-8 explicitly supports patient names containing non-ASCII letters.
    """
    import pydicom
    from pydicom.errors import InvalidDicomError
    for path in Path(folder).rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path))
        except InvalidDicomError:
            continue
        if getattr(ds, "Modality", "") not in ("CT", "RTSTRUCT"):
            continue
        ds.SpecificCharacterSet = "ISO_IR 192"
        for key, value in metadata.items():
            setattr(ds, key, value)
        if ds.Modality == "CT":
            ds.ImageType = ["DERIVED", "SECONDARY"]
            ds.DerivationDescription = "MRI-derived synthetic CT; Synth-O-Matic " + VERSION
        ds.save_as(str(path))


def export_dicom(workflow, folder, metadata):
    """Export only the committed snapshot; clean temporary MRML objects on error."""
    import DicomRtImportExportPlugin
    result = workflow.result
    nodes, patient = [], None
    sh = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    constants = slicer.vtkMRMLSubjectHierarchyConstants
    try:
        ct = make_volume(workflow.input, result["data"], "sCT_EXPORT_TEMP")
        nodes.append(ct)
        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "sCT_STRUCT_TEMP")
        nodes.append(seg)
        seg.CreateDefaultDisplayNodes()
        seg.SetReferenceImageGeometryParameterFromVolumeNode(ct)
        for key, name in SETTINGS["segments"].items():
            sid = seg.GetSegmentation().AddEmptySegment("", name)
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                result["masks"][key].astype(np.uint8), seg, sid, ct)
        patient = sh.CreateSubjectItem(sh.GetSceneItemID(), metadata["PatientID"])
        study = sh.CreateStudyItem(patient, "MRI-based synthetic CT")
        # SlicerRT reads the study UID via GetItemUID, not a StudyInstanceUID attribute.
        study_uid = "2.25." + str(uuid.uuid4().int)
        sh.SetItemUID(study, constants.GetDICOMUIDName(), study_uid)
        sh.SetItemAttribute(study, constants.GetDICOMStudyIDTagName(), metadata["StudyID"])
        exporter = DicomRtImportExportPlugin.DicomRtImportExportPluginClass()
        exportables = []
        for node, modality, description, number in (
                (ct, "CT", "Synth-O-Matic synthetic CT", "301"),
                (seg, "RTSTRUCT", "sCT effective conversion masks", "302")):
            item = sh.GetItemByDataNode(node)
            sh.SetItemParent(item, study)
            candidates = [e for e in exporter.examineForExport(item) if e.tag("Modality") == modality]
            if len(candidates) != 1:
                raise RuntimeError(f"SlicerRT did not provide one {modality} exporter.")
            e = candidates[0]
            e.directory = str(folder)
            for key, value in metadata.items():
                getter = getattr(constants, "GetDICOM" + key + "TagName", None)
                e.setTag(getter() if getter else key, value)
            e.setTag("SeriesDescription", description)
            e.setTag("SeriesNumber", number)
            exportables.append(e)
        # Export CT and RTSTRUCT together. SlicerRT owns linked series/frame/SOP UIDs.
        error = exporter.export(exportables)
        if error:
            raise RuntimeError(error)
        finalize_metadata(folder, metadata)
        report = verify_dicom(folder, result)
        with open(Path(folder)/"conversion_parameters.json", "w", encoding="utf-8") as f:
            json.dump({**result["info"], "verification": report}, f, indent=2, allow_nan=False)
        return report
    finally:
        remove_nodes(nodes)
        if patient is not None:
            sh.RemoveItem(patient, False, True)


# ---- Dialog: collect user intent, then call calculation/adapter functions ---
class SynthOMatic:
    def __init__(self):
        self.workflow = Workflow()
        self.fields, self.target_fields, self.coefficient_fields = {}, [], []
        self.plot_nodes, self.hist_nodes = {}, {}
        self.bone_mask_hash = None
        self.qa_node = None
        self.dialog = qt.QDialog(slicer.util.mainWindow())
        self.dialog.setWindowTitle("Synth-O-Matic " + VERSION)
        outer = qt.QVBoxLayout(self.dialog)
        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        content = qt.QWidget()
        self.layout = qt.QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        self.build_ui()
        self.dialog.resize(580, 900)

    def label(self, text):
        label = qt.QLabel(text)
        label.setWordWrap(True)
        self.layout.addWidget(label)
        return label

    def button(self, text, callback):
        button = qt.QPushButton(text)
        button.connect("clicked()", lambda: self.safe(callback))
        self.layout.addWidget(button)
        return button

    def numeric(self, form, label, value, minimum, maximum, step, decimals=2):
        widget = qt.QDoubleSpinBox()
        widget.setDecimals(decimals)
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.connect("valueChanged(double)", self.mark_dirty)
        form.addRow(label, widget)
        return widget

    def safe(self, callback):
        try:
            callback()
        except Exception as error:
            LOG.exception("Synth-O-Matic operation failed")
            self.status.setText("Operation failed: " + str(error))
            slicer.util.errorDisplay(str(error))

    def mark_dirty(self, *args):
        if hasattr(self, "status"):
            self.status.setText("Settings changed. Update preview to apply them. Export always recalculates.")

    def build_ui(self):
        self.label("<b>SOFT-TISSUE CONVERSION</b>")
        form = qt.QFormLayout()
        self.layout.addLayout(form)
        for key, title in (("urine_max", "Urine maximum"), ("muscle_min", "Muscle minimum"),
                           ("muscle_max", "Muscle maximum"), ("fat_min", "Fat minimum")):
            self.fields[key] = self.numeric(form, title, SETTINGS[key], -10000, 100000, 10)
        self.statistics = self.label("")
        self.button("Show soft-tissue MRI histogram", self.show_histogram)
        self.label("<b>BONE CONVERSION</b><br>HU = exp(a + b*x + c*x*x) - 1")
        form = qt.QFormLayout()
        self.layout.addLayout(form)
        for title, value in zip(("HU at minimum MRI", "HU at P50 MRI", "HU at maximum MRI"),
                                SETTINGS["bone_targets"]):
            self.target_fields.append(self.numeric(form, title, value, 0, SETTINGS["bone_max_hu"], 25))
        self.label("Changing target HU values requires 'Fit MIN / P50 / MAX'.")
        form = qt.QFormLayout()
        self.layout.addLayout(form)
        for name, step in zip(("a", "b", "c"), (0.01, 0.0001, 0.000001)):
            self.coefficient_fields.append(self.numeric(form, name, 0, -1e6, 1e6, step, 12))
        self.button("Fit MIN / P50 / MAX", self.refit)
        self.fit_status = self.label("")
        self.allow_nonmonotone = qt.QCheckBox("Allow export of a non-monotone bone curve")
        self.layout.addWidget(self.allow_nonmonotone)
        self.button("Update sCT preview + bone curve", self.update_preview)
        self.button("Show bone conversion curve", self.show_curve)
        self.button("Show unassigned BODY voxels", self.show_unassigned)
        self.button("Run N4 again (after MRI / BODY changes)", self.run_n4)
        self.label("<b>DISPLAY WINDOW</b>")
        form = qt.QFormLayout()
        self.layout.addLayout(form)
        self.window_min = self.numeric(form, "Minimum HU", -300, -5000, 10000, 50)
        self.window_max = self.numeric(form, "Maximum HU", 300, -5000, 10000, 50)
        self.button("Apply custom window", self.apply_window)
        self.button("Soft tissue window (-300 ... 300)", lambda: self.set_window(-300, 300))
        self.button("Bone window (-500 ... 1500)", lambda: self.set_window(-500, 1500))
        self.button("Adjust window / level with mouse", self.mouse_window)
        self.label("<b>IMAGE VIEW</b>")
        for text, key in (("sCT preview", "output"), ("N4-corrected MRI", "n4"),
                          ("Original MRI", "input"), ("Bias field", "bias")):
            self.button("Show " + text, lambda key=key: self.show_volume(key))
        self.label("<b>DICOM EXPORT</b><br>Recalculates from current settings and exports the effective masks. Requires SlicerRT.")
        form = qt.QFormLayout()
        self.layout.addLayout(form)
        self.metadata_fields = {}
        for key, title, tag, fallback in (
                ("PatientName", "Patient name", "0010,0010", "sCT^Patient"),
                ("PatientID", "Patient ID", "0010,0020", "sCT001"),
                ("StudyID", "Study ID", "0020,0010", "SCT001")):
            edit = qt.QLineEdit()
            edit.setText(dicom_tag(self.workflow.input, tag, fallback))
            form.addRow(title, edit)
            self.metadata_fields[key] = edit
        self.button("Export sCT + RTSTRUCT to DICOM", self.export)
        self.status = self.label("Starting...")
        self.button("Close", self.dialog.close)

    def parameters(self):
        p = dict(SETTINGS)
        p.update({k: float(w.value) for k, w in self.fields.items()})
        p["bone_targets"] = [float(w.value) for w in self.target_fields]
        validate_parameters(p)
        return p

    def coefficients(self):
        return np.array([w.value for w in self.coefficient_fields], dtype=np.float64)

    def sync_bone(self, masks, force=False):
        mask_hash = fingerprint(masks["bone"])
        if force or mask_hash != self.bone_mask_hash:
            anchors = bone_anchors(self.workflow.image[masks["bone"]].astype(np.float64))
            coefficients = fit_bone(anchors, [w.value for w in self.target_fields])
            for widget, value in zip(self.coefficient_fields, coefficients):
                widget.setValue(float(value))
            self.bone_mask_hash = mask_hash
            return True
        return False

    def run_n4(self):
        self.status.setText("Running N4...")
        masks = self.workflow.run_n4()
        self.sync_bone(masks, force=True)
        self.update_preview()
        self.set_window(-300, 300)

    def refit(self):
        masks = self.workflow.fresh_masks()
        self.sync_bone(masks, force=True)
        self.update_preview()

    def update_preview(self):
        masks = self.workflow.fresh_masks()
        refitted = self.sync_bone(masks)
        result = self.workflow.commit(masks, self.parameters(), self.coefficients())
        self.update_statistics(masks)
        self.draw_curve(result)
        if self.qa_node is not None and self.qa_node.GetScene() is not None:
            self.refresh_unassigned(result)
        self.show_volume("output")
        info = result["info"]
        self.status.setText(
            f"Preview updated. Unassigned BODY: {info['unassigned_voxels']} voxels "
            f"({info['unassigned_cm3']:.2f} cm3), retained at {SETTINGS['outside_hu']} HU."
            + (" Bone mask changed: MIN/P50/MAX refitted." if refitted else ""))
        return result

    def update_statistics(self, masks):
        original = np.percentile(self.workflow.original[masks["soft"]], PERCENTILES)
        corrected = np.percentile(self.workflow.image[masks["soft"]], PERCENTILES)
        rows = "\n".join(f"P{p:02d}: {a:.1f} -> {b:.1f}" for p, a, b in zip(PERCENTILES, original, corrected))
        bone = self.workflow.image[masks["bone"]].astype(np.float64)
        values = np.percentile(bone, [0, 1, 5, 50, 95, 99, 100])
        stats = " | ".join(f"{k}: {v:.1f}" for k, v in zip(("MIN", "P01", "P05", "P50", "P95", "P99", "MAX"), values))
        self.statistics.setText("SOFT TISSUE: original -> N4\n" + rows + "\nBONE (N4):\n" + stats)

    def draw_curve(self, result):
        info = result["info"]
        anchors, coefficients = info["bone_anchors"], info["coefficients"]
        x = np.linspace(anchors[0], anchors[2], 500)
        y = bone_hu(x, coefficients, info["parameters"])
        self.chart = slicer.util.plot(np.column_stack([x, y]), xColumnIndex=0,
            columnNames=["MRI intensity", "HU"], title="Bone MRI-to-HU curve",
            show=False, nodes=self.plot_nodes)
        self.chart.SetXAxisTitle("N4 MRI intensity")
        self.chart.SetYAxisTitle("HU")
        series = self.chart.GetNthPlotSeriesNode(0)
        series.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        series.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
        series.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid)
        actual = bone_hu(np.array(anchors), coefficients, info["parameters"])
        text = "\n".join(f"{name} ({x:.2f}) -> {hu:.2f} HU" for name, x, hu in zip(("MIN", "P50", "MAX"), anchors, actual))
        self.fit_status.setText(text + ("\nCurve is non-increasing." if info["monotone"]
                                      else "\nWARNING: bone curve is not monotone. Review before export."))

    def show_curve(self):
        self.update_preview()
        slicer.modules.plots.logic().ShowChartInLayout(self.chart)

    def show_histogram(self):
        masks = self.workflow.fresh_masks()
        values = self.workflow.image[masks["soft"]]
        lo, hi = np.percentile(values, [0.5, 99.5])
        if lo == hi:
            lo, hi = lo-0.5, hi+0.5
        histogram = np.histogram(values, bins=200, range=(lo, hi))
        chart = slicer.util.plot(histogram, xColumnIndex=1, title="N4 soft-tissue histogram",
                                 show=False, nodes=self.hist_nodes)
        chart.SetXAxisTitle("N4 MRI intensity")
        chart.SetYAxisTitle("Voxel count")
        slicer.modules.plots.logic().ShowChartInLayout(chart)

    def show_unassigned(self):
        self.refresh_unassigned(self.update_preview())

    def refresh_unassigned(self, result):
        masks = result["masks"]
        if self.qa_node is None or self.qa_node.GetScene() is None:
            self.qa_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "sCT_QA")
            self.qa_node.CreateDefaultDisplayNodes()
            self.qa_id = self.qa_node.GetSegmentation().AddEmptySegment("", "Unassigned BODY", [1, 0, 1])
        self.qa_node.SetReferenceImageGeometryParameterFromVolumeNode(self.workflow.output)
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            (masks["body"] & ~(masks["soft"] | masks["bone"])).astype(np.uint8),
            self.qa_node, self.qa_id, self.workflow.output)
        self.qa_node.GetDisplayNode().SetVisibility(True)

    def show_volume(self, key):
        node = getattr(self.workflow, key)
        if node is None or node.GetScene() is None:
            raise ValueError("Image is unavailable. Run N4/update preview first.")
        slicer.util.setSliceViewerLayers(background=node)
        node.GetDisplayNode().SetAutoWindowLevel(key != "output")

    def apply_window(self):
        lo, hi = float(self.window_min.value), float(self.window_max.value)
        if lo >= hi:
            raise ValueError("Window minimum must be smaller than maximum.")
        self.show_volume("output")
        self.workflow.output.GetDisplayNode().SetWindowLevelMinMax(lo, hi)

    def set_window(self, lo, hi):
        self.window_min.setValue(lo)
        self.window_max.setValue(hi)
        self.apply_window()

    def mouse_window(self):
        self.show_volume("output")
        slicer.app.applicationLogic().GetInteractionNode().SetCurrentInteractionMode(
            slicer.vtkMRMLInteractionNode.AdjustWindowLevel)

    def export(self):
        if not hasattr(slicer.modules, "dicomrtimportexport"):
            raise RuntimeError("Install SlicerRT using Extension Manager and restart Slicer.")
        import pydicom  # Verify dependency before creating any files.
        result = self.update_preview()
        if not result["info"]["monotone"] and not self.allow_nonmonotone.checked:
            raise ValueError("Bone curve is not monotone. Adjust the fit or explicitly allow export.")
        metadata = {k: str(w.text).strip() for k, w in self.metadata_fields.items()}
        if not all(metadata.values()):
            raise ValueError("Patient name, Patient ID and Study ID must not be empty.")
        now = datetime.datetime.now()
        metadata.update({"PatientSex": dicom_tag(self.workflow.input, "0010,0040"),
                         "PatientBirthDate": dicom_tag(self.workflow.input, "0010,0030"),
                         "StudyDate": now.strftime("%Y%m%d"), "StudyTime": now.strftime("%H%M%S"),
                         "StudyDescription": "MRI-based synthetic CT"})
        selected = qt.QFileDialog.getExistingDirectory(self.dialog, "Select DICOM output folder")
        if not selected:
            return
        folder = Path(str(selected))/("sCT_DICOM_" + now.strftime("%Y%m%d_%H%M%S_%f"))
        folder.mkdir(exist_ok=False)
        try:
            # Directory selection runs a nested event loop: recheck the source and recompute.
            result = self.update_preview()
            if not result["info"]["monotone"] and not self.allow_nonmonotone.checked:
                raise ValueError("Bone curve changed and has not been accepted for export.")
            report = export_dicom(self.workflow, folder, metadata)
        except Exception:
            (folder/"EXPORT_FAILED.txt").write_text(
                "Export or verification failed. Files may be incomplete; see Slicer's Python console.\n",
                encoding="utf-8")
            raise
        self.status.setText(f"Exported and verified {report['ct_slices']} CT slices + RTSTRUCT:\n{folder}")
        slicer.util.infoDisplay("DICOM export complete. CT HU, geometry and image references verified.\n"
                               "RTSTRUCT contour accuracy still requires visual review.\n\n" + str(folder))

    def start(self):
        self.dialog.show()
        self.safe(self.run_n4)


# Keep the instance alive. Each paste uses a fresh namespace, preventing old
# callbacks from resolving newly rebound globals. Results remain visible.
_previous = getattr(slicer, "_synth_o_matic", None)
if _previous is not None:
    _previous.dialog.close()
slicer._synth_o_matic = SynthOMatic()
slicer._synth_o_matic.start()
''', '<SynthOMatic>', 'exec'), {"__name__": "__synth_o_matic__"})
