"""Public API for standalone 3D object position extraction."""

from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw

from .config import Object3DExtractionConfig
from .depth_module import DepthModule
from .detection_module import DetectionModule
from .orientation_module import OrientationModule
from .utils import bbox_wh, load_rgb_image, save_debug_visuals, validate_object_names


class Object3DLocator:
    def __init__(
        self,
        config: Optional[Union[Object3DExtractionConfig, Dict[str, Any]]] = None,
        device: str = "cuda",
        detection_module: Optional[Any] = None,
        depth_module: Optional[Any] = None,
        orientation_module: Optional[Any] = None,
        vlm_model: Optional[Any] = None,
    ):
        self.config = _coerce_config(config)
        self.device = device
        self.detection_module = detection_module or DetectionModule(self.config, device=device)
        self.depth_module = depth_module or DepthModule(self.config, device=device)
        self.orientation_module = orientation_module or OrientationModule(self.config, device=device)
        self.vlm_model = vlm_model
        self.last_3d_visualization_paths = None

    def extract(
        self,
        image,
        object_names: List[str],
        visualize: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        question: str = "",
        answer: str = "",
        question_type: str = None,
        forced_candidate_selections: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict[str, object]]:
        image_pil = load_rgb_image(image)
        names = validate_object_names(object_names)
        forced_candidate_selections = forced_candidate_selections or {}

        results = {}  # type: Dict[str, Dict[str, object]]
        selected_boxes = {}  # type: Dict[str, List[int]]
        depth = None

        for object_name in names:
            box2d = None
            mask = None
            sam_mask = None
            mask_used_for_3d = None
            mask_fallback_reason = None
            prompts_tried = detection_prompts_for_object(object_name)
            count_target = is_visual_count_target(question_type, question, object_name)
            try:
                candidates = self._collect_detection_candidates(
                    image_pil,
                    object_name,
                    prompts_tried,
                    is_count_target=count_target,
                )
                if not candidates:
                    results[object_name] = {
                        "error": f"No detection for object: {object_name}",
                        "prompts_tried": prompts_tried,
                    }
                    continue

                if count_target:
                    instance_results, instance_names, depth, filter_debug = self._extract_count_instances(
                        image_pil=image_pil,
                        object_name=object_name,
                        candidates=candidates,
                        selected_boxes=selected_boxes,
                        depth=depth,
                        visualize=visualize,
                        save_dir=save_dir,
                    )
                    results[object_name] = {
                        "error": "Count target expanded into indexed instances",
                        "counting_target": True,
                        "counting_instances": instance_names,
                        "counting_instance_count": len(instance_names),
                        "counting_filter_summary": filter_debug.get("summary"),
                        "counting_rejected_candidates": filter_debug.get("rejected_candidates", []),
                        "prompts_tried": prompts_tried,
                        "candidates_considered": summarize_candidates(candidates),
                    }
                    results.update(instance_results)
                    continue

                best_detection = self._select_detection(
                    image_pil=image_pil,
                    object_name=object_name,
                    candidates=candidates,
                    selected_boxes=selected_boxes,
                    save_dir=save_dir if visualize else None,
                    question=question,
                    forced_candidate_selections=forced_candidate_selections,
                )
                box2d = [int(v) for v in best_detection["box2d"]]
                sam_mask = self.detection_module.run_segmentation(image_pil, box2d)
                if self.config.mask_fallback_mode == "auto":
                    mask, mask_fallback_reason = maybe_fallback_mask(image_pil, object_name, box2d, sam_mask)
                else:
                    mask = sam_mask
                    mask_fallback_reason = None
                mask_used_for_3d = "fallback" if mask_fallback_reason else "sam"
                sam_mask_area = int((sam_mask > 0.5).sum())
                mask_area = int((mask > 0.5).sum())
                if mask_area < self.config.min_mask_area:
                    results[object_name] = {
                        "error": f"Segmentation mask too small: {mask_area} < {self.config.min_mask_area}",
                        "box2d": box2d,
                        "bbox_wh": bbox_wh(box2d),
                        "mask_area": mask_area,
                        "sam_mask_area": sam_mask_area,
                        "final_mask_area": mask_area,
                        "mask_used_for_3d": mask_used_for_3d,
                        "prompt_used": best_detection.get("prompt"),
                        "candidate_rank_reason": best_detection.get("rank_reason"),
                        "rule_selected_index": best_detection.get("rule_selected_index"),
                        "vlm_selected_index": best_detection.get("vlm_selected_index"),
                        "final_selected_index": best_detection.get("final_selected_index"),
                        "selection_decision": best_detection.get("selection_decision"),
                        "selection_reject_reason": best_detection.get("selection_reject_reason"),
                        "vlm_response": best_detection.get("vlm_response"),
                        "vlm_candidate_scores": best_detection.get("vlm_candidate_scores"),
                        "mask_fallback_reason": mask_fallback_reason,
                        "candidate_overlay_path": best_detection.get("candidate_overlay_path"),
                        "candidate_grid_path": best_detection.get("candidate_grid_path"),
                        "selected_overlay_path": best_detection.get("selected_overlay_path"),
                        "candidates_considered": summarize_candidates(best_detection.get("_ranked_candidates", candidates)),
                    }
                    continue

                if depth is None:
                    depth = self.depth_module.run_depth_estimation(image_pil)

                unprojected = self.depth_module.unproject_to_3D(image_pil, depth, mask)
                orientation = self.orientation_module.run_orientation_estimation(image_pil, box2d)
                selected_boxes[object_name] = box2d
                results[object_name] = {
                    "position": unprojected["position"],
                    "orientation": orientation["orientation"],
                    "orientation_angles": orientation["orientation_angles"],
                    "box3d_size": unprojected["box3d_size"],
                    "box3d_center": unprojected["box3d_center"],
                    "box3d_min": unprojected["box3d_min"],
                    "box3d_max": unprojected["box3d_max"],
                    "bbox_wh": bbox_wh(box2d),
                    "box2d": box2d,
                    "mask_area": int(unprojected["mask_area"]),
                    "sam_mask_area": sam_mask_area,
                    "final_mask_area": int(unprojected["mask_area"]),
                    "mask_used_for_3d": mask_used_for_3d,
                    "depth_mode": float(unprojected["depth_mode"]),
                    "num_points": int(unprojected["num_points"]),
                    "prompt_used": best_detection.get("prompt"),
                    "candidate_rank_reason": best_detection.get("rank_reason"),
                    "rule_selected_index": best_detection.get("rule_selected_index"),
                    "vlm_selected_index": best_detection.get("vlm_selected_index"),
                    "final_selected_index": best_detection.get("final_selected_index"),
                    "selection_decision": best_detection.get("selection_decision"),
                    "selection_reject_reason": best_detection.get("selection_reject_reason"),
                    "vlm_response": best_detection.get("vlm_response"),
                    "vlm_candidate_scores": best_detection.get("vlm_candidate_scores"),
                    "mask_fallback_reason": mask_fallback_reason,
                    "candidate_overlay_path": best_detection.get("candidate_overlay_path"),
                    "candidate_grid_path": best_detection.get("candidate_grid_path"),
                    "selected_overlay_path": best_detection.get("selected_overlay_path"),
                    "overlap_warnings": overlap_warnings(object_name, box2d, selected_boxes),
                    "candidates_considered": summarize_candidates(best_detection.get("_ranked_candidates", candidates)),
                }

                if "score" in best_detection:
                    results[object_name]["score"] = float(best_detection["score"])

            except Exception as exc:
                error_result = {"error": str(exc)}  # type: Dict[str, object]
                if box2d is not None:
                    error_result["box2d"] = box2d
                    error_result["bbox_wh"] = bbox_wh(box2d)
                if sam_mask is not None:
                    error_result["sam_mask_area"] = int((sam_mask > 0.5).sum())
                if mask is not None:
                    error_result["mask_area"] = int((mask > 0.5).sum())
                    error_result["final_mask_area"] = int((mask > 0.5).sum())
                if mask_used_for_3d is not None:
                    error_result["mask_used_for_3d"] = mask_used_for_3d
                if mask_fallback_reason is not None:
                    error_result["mask_fallback_reason"] = mask_fallback_reason
                results[object_name] = error_result
            finally:
                if visualize and (box2d is not None or mask is not None or sam_mask is not None):
                    save_debug_visuals(
                        image_pil,
                        object_name,
                        box2d,
                        mask,
                        save_dir,
                        sam_mask=sam_mask,
                        mask_used_for_3d=mask_used_for_3d,
                    )

        if visualize and save_dir is not None:
            panel_question = question or "Objects: " + ", ".join(names)
            try:
                from .visualize_3d_aabb import visualize_3d_debug_panel

                self.last_3d_visualization_paths = visualize_3d_debug_panel(
                    image_path_or_pil=image_pil,
                    question=panel_question,
                    results=results,
                    save_dir=str(save_dir),
                    prefix="3d_debug",
                    answer=answer,
                )
            except Exception as exc:
                print(f"[WARN] Failed to create 3D debug panel: {exc}")

        return results

    def _collect_detection_candidates(
        self,
        image_pil: Image.Image,
        object_name: str,
        prompts: List[str],
        is_count_target: bool = False,
    ) -> List[Dict[str, object]]:
        per_prompt_max_candidates = None
        if is_count_target:
            per_prompt_max_candidates = max(
                int(self.config.detection.count_max_instances),
                int(self.config.detection.num_candidates) * int(self.config.detection.count_candidate_multiplier),
            )
        candidates = []
        for prompt in prompts:
            detections = self.detection_module.detect(image_pil, prompt, max_candidates=per_prompt_max_candidates)
            for det in detections:
                candidate = dict(det)
                candidate["prompt"] = prompt
                candidate["object_name"] = object_name
                candidate["box2d"] = [int(v) for v in candidate["box2d"]]
                candidates.append(candidate)
        candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        if is_count_target:
            candidate_limit = max(
                int(self.config.detection.count_max_instances),
                int(self.config.detection.num_candidates) * len(prompts) * int(self.config.detection.count_candidate_multiplier),
            )
        else:
            candidate_limit = int(self.config.detection.num_candidates) * len(prompts)
        candidates = candidates[: max(1, candidate_limit)]
        return rank_candidates_for_object(image_pil, object_name, candidates, {})

    def _extract_count_instances(
        self,
        image_pil: Image.Image,
        object_name: str,
        candidates: List[Dict[str, object]],
        selected_boxes: Dict[str, List[int]],
        depth,
        visualize: bool,
        save_dir: Optional[Union[str, Path]],
    ) -> Tuple[Dict[str, Dict[str, object]], List[str], Any, Dict[str, object]]:
        instances = {}  # type: Dict[str, Dict[str, object]]
        instance_names = []  # type: List[str]
        selected_candidates, filter_debug = filter_count_instance_candidates(
            candidates,
            max_instances=max(1, self.config.detection.count_max_instances),
            image_size=image_pil.size,
        )

        for idx, detection in enumerate(selected_candidates, start=1):
            instance_name = make_count_instance_name(object_name, idx)
            box2d = [int(v) for v in detection["box2d"]]
            sam_mask = self.detection_module.run_segmentation(image_pil, box2d)
            if self.config.mask_fallback_mode == "auto":
                mask, mask_fallback_reason = maybe_fallback_mask(image_pil, object_name, box2d, sam_mask)
            else:
                mask = sam_mask
                mask_fallback_reason = None
            mask_used_for_3d = "fallback" if mask_fallback_reason else "sam"
            sam_mask_area = int((sam_mask > 0.5).sum())
            mask_area = int((mask > 0.5).sum())

            if mask_area < self.config.min_mask_area:
                instances[instance_name] = {
                    "error": f"Segmentation mask too small: {mask_area} < {self.config.min_mask_area}",
                    "counting_source_object": object_name,
                    "box2d": box2d,
                    "bbox_wh": bbox_wh(box2d),
                    "mask_area": mask_area,
                    "sam_mask_area": sam_mask_area,
                    "final_mask_area": mask_area,
                    "mask_used_for_3d": mask_used_for_3d,
                    "prompt_used": detection.get("prompt"),
                    "candidate_rank_reason": "count_instance_ranker",
                    "mask_fallback_reason": mask_fallback_reason,
                    "candidates_considered": summarize_candidates([detection]),
                }
                continue

            if depth is None:
                depth = self.depth_module.run_depth_estimation(image_pil)

            unprojected = self.depth_module.unproject_to_3D(image_pil, depth, mask)
            orientation = self.orientation_module.run_orientation_estimation(image_pil, box2d)
            selected_boxes[instance_name] = box2d
            instance_names.append(instance_name)
            instances[instance_name] = {
                "position": unprojected["position"],
                "orientation": orientation["orientation"],
                "orientation_angles": orientation["orientation_angles"],
                "box3d_size": unprojected["box3d_size"],
                "box3d_center": unprojected["box3d_center"],
                "box3d_min": unprojected["box3d_min"],
                "box3d_max": unprojected["box3d_max"],
                "bbox_wh": bbox_wh(box2d),
                "box2d": box2d,
                "mask_area": int(unprojected["mask_area"]),
                "sam_mask_area": sam_mask_area,
                "final_mask_area": int(unprojected["mask_area"]),
                "mask_used_for_3d": mask_used_for_3d,
                "depth_mode": float(unprojected["depth_mode"]),
                "num_points": int(unprojected["num_points"]),
                "prompt_used": detection.get("prompt"),
                "candidate_rank_reason": "count_instance_ranker",
                "rule_selected_index": detection.get("candidate_index"),
                "vlm_selected_index": None,
                "final_selected_index": detection.get("candidate_index"),
                "selection_decision": "count_instance",
                "selection_reject_reason": None,
                "vlm_response": None,
                "mask_fallback_reason": mask_fallback_reason,
                "overlap_warnings": overlap_warnings(instance_name, box2d, selected_boxes),
                "candidates_considered": summarize_candidates([detection]),
                "counting_source_object": object_name,
                "counting_instance_id": idx,
            }
            if "score" in detection:
                instances[instance_name]["score"] = float(detection["score"])

            if visualize and (box2d is not None or mask is not None or sam_mask is not None):
                save_debug_visuals(
                    image_pil,
                    instance_name,
                    box2d,
                    mask,
                    save_dir,
                    sam_mask=sam_mask,
                    mask_used_for_3d=mask_used_for_3d,
                )

        return instances, instance_names, depth, filter_debug


    def _select_detection(
        self,
        image_pil: Image.Image,
        object_name: str,
        candidates: List[Dict[str, object]],
        selected_boxes: Dict[str, List[int]],
        save_dir: Optional[Union[str, Path]],
        question: str,
        forced_candidate_selections: Optional[Dict[str, int]] = None,
    ) -> Dict[str, object]:
        rule_ranked = rank_candidates_for_object(image_pil, object_name, candidates, selected_boxes)
        rule_selected = dict(rule_ranked[0])
        rule_selected["_ranked_candidates"] = rule_ranked
        rule_selected["rule_selected_index"] = 0
        rule_selected["vlm_selected_index"] = None
        rule_selected["final_selected_index"] = 0
        rule_selected["selection_decision"] = "rule_ranker"
        rule_selected["selection_reject_reason"] = None
        forced_reject_reason = None

        forced_candidate_selections = forced_candidate_selections or {}
        if object_name in forced_candidate_selections:
            try:
                forced_index = int(forced_candidate_selections[object_name])
            except (TypeError, ValueError):
                forced_index = None
            if forced_index is not None:
                for candidate in rule_ranked:
                    if int(candidate.get("candidate_index", -1)) == forced_index:
                        selected = dict(candidate)
                        selected["_ranked_candidates"] = rule_ranked
                        selected["rule_selected_index"] = 0
                        selected["vlm_selected_index"] = forced_index
                        selected["final_selected_index"] = forced_index
                        selected["selection_decision"] = "forced_vlm_candidate_scoring"
                        selected["selection_reject_reason"] = None
                        selected["rank_reason"] = "forced_vlm_candidate_scoring"
                        if save_dir is not None:
                            save_candidate_grid(image_pil, object_name, rule_ranked, save_dir)
                            selected["candidate_overlay_path"] = str(
                                save_candidate_overlay(image_pil, object_name, rule_ranked, save_dir)
                            )
                        return selected
            forced_reject_reason = "invalid_forced_candidate"
            rule_selected["selection_decision"] = "rule_ranker_forced_invalid"
            rule_selected["selection_reject_reason"] = forced_reject_reason

        if self.config.detection.use_vlm_refinement and self.vlm_model is not None:
            selected = select_candidate_with_vlm(
                self.vlm_model,
                image_pil,
                object_name,
                rule_ranked,
                save_dir=save_dir,
                question=question,
                selected_boxes=selected_boxes,
            )
            if selected is None:
                rule_selected["selection_decision"] = "rule_ranker_vlm_invalid"
                rule_selected["selection_reject_reason"] = forced_reject_reason or "invalid_vlm_response"
            else:
                allowed, reject_reason = validate_vlm_selection(object_name, selected, rule_ranked, image_size=image_pil.size)
                rule_selected["vlm_selected_index"] = selected.get("candidate_index")
                if allowed:
                    selected = dict(selected)
                    selected["_ranked_candidates"] = rule_ranked
                    selected["rule_selected_index"] = 0
                    selected["vlm_selected_index"] = selected.get("candidate_index")
                    selected["final_selected_index"] = selected.get("candidate_index")
                    selected["selection_decision"] = "vlm_refinement"
                    selected["selection_reject_reason"] = None
                    selected["rank_reason"] = "vlm_refinement"
                    return selected
                rule_selected["selection_decision"] = "rule_ranker_vlm_rejected"
                rule_selected["selection_reject_reason"] = reject_reason

            if save_dir is not None:
                rule_selected["candidate_overlay_path"] = str(
                    save_candidate_overlay(image_pil, object_name, rule_ranked, save_dir)
                )

        rule_selected["rank_reason"] = "rule_ranker"
        if save_dir is not None and "candidate_overlay_path" not in rule_selected:
            save_candidate_grid(image_pil, object_name, rule_ranked, save_dir)
            overlay_path = save_candidate_overlay(image_pil, object_name, rule_ranked, save_dir)
            rule_selected["candidate_overlay_path"] = str(overlay_path)
        return rule_selected


def detection_prompts_for_object(object_name: str) -> List[str]:
    original = str(object_name).strip().lower()
    context = parse_object_relation_context(original)
    target = context["target_phrase"]
    prompt_target = _remove_count_quantifier(target or original)
    prompts = [prompt_target]

    base = _remove_relation_words(prompt_target)
    if base and base not in prompts:
        prompts.append(base)

    without_color = _remove_leading_color(base)
    if without_color != base:
        prompts.append(without_color)

    if "coffee table" in target:
        if "white" in target:
            prompts.append("white table")
        prompts.extend(["coffee table", "table"])
    elif " table" in target or target == "table":
        if is_shape_object(target):
            if "circular" in target:
                prompts.append(target.replace("circular", "round"))
            if "round" in target:
                prompts.append(target.replace("round", "circular"))
            prompts.append("table")
        if is_color_object(target):
            prompts.append(target)

    if is_transparency_object(target):
        prompts.extend(_transparency_synonym_prompts(target))
        transparency_base = _remove_leading_transparency(target)
        if transparency_base and transparency_base != target:
            prompts.append(transparency_base)
    if is_material_object(target) and "table" in target:
        prompts.extend(["transparent table", "glass coffee table", "clear table"])
    if target in {"tv", "television"}:
        prompts.extend(["tv", "television"])
    return _dedupe_preserve_order([prompt for prompt in prompts if prompt])


def rank_detection_candidates(
    image_pil: Image.Image,
    object_name: str,
    candidates: List[Dict[str, object]],
    selected_boxes: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, object]:
    ranked = rank_candidates_for_object(image_pil, object_name, candidates, selected_boxes or {})
    return ranked[0]


def rank_candidates_for_object(
    image_pil: Image.Image,
    object_name: str,
    candidates: List[Dict[str, object]],
    selected_boxes: Optional[Dict[str, List[int]]] = None,
) -> List[Dict[str, object]]:
    selected_boxes = selected_boxes or {}
    width, height = image_pil.size
    relation = relation_modifier(object_name)
    scored = []
    for idx, candidate in enumerate(candidates):
        score, reasons = score_detection_candidate(image_pil, object_name, candidate, selected_boxes)
        item = dict(candidate)
        item["candidate_index"] = idx
        item["rank_score"] = float(score)
        item["rank_reasons"] = reasons
        scored.append((score, idx, item))

    if relation in {"leftmost", "rightmost", "center", "topmost", "bottommost", "closest", "furthest"}:
        selected = select_by_relation(scored, relation, width, height)
        rest = [item for _, _, item in sorted(scored, key=lambda value: value[0], reverse=True) if item is not selected]
        return [selected] + rest

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[2] for item in scored]


def score_detection_candidate(
    image_pil: Image.Image,
    object_name: str,
    candidate: Dict[str, object],
    selected_boxes: Dict[str, List[int]],
) -> Tuple[float, List[str]]:
    width, height = image_pil.size
    box = candidate["box2d"]
    area_ratio = box_area(box) / float(max(1, width * height))
    box_w = max(0, int(box[2]) - int(box[0]))
    box_h = max(0, int(box[3]) - int(box[1]))
    width_ratio = box_w / float(max(1, width))
    height_ratio = box_h / float(max(1, height))
    cx, cy = box_center(box)
    score = float(candidate.get("score", 0.0))
    reasons = ["dino_score"]
    area_kind = is_area_object(object_name)

    prompt = str(candidate.get("prompt", "")).lower()
    original = object_name.lower()
    context = parse_object_relation_context(original)
    target = context["target_phrase"]
    if prompt == original:
        score += 0.18
        reasons.append("exact_prompt")
    elif prompt == target:
        score += 0.16
        reasons.append("target_prompt")
    elif prompt and (prompt in original or prompt in target):
        score += 0.08
        reasons.append("specific_prompt")
    if is_color_object(target) and prompt_has_color(prompt, target):
        score += 0.12
        reasons.append("color_prompt")
    if is_shape_object(target) and prompt_has_shape(prompt, target):
        score += 0.12
        reasons.append("shape_prompt")
    if is_material_object(target) and prompt_has_material(prompt, target):
        score += 0.14
        reasons.append("material_prompt")

    if area_kind:
        if area_ratio > 0.15:
            score += 0.25
            reasons.append("large_area_object")
        if width_ratio > 0.45:
            score += 0.15
            reasons.append("wide_area_object")
        if cy > height * 0.45:
            score += 0.15
            reasons.append("low_area_object")
        if area_ratio < 0.08:
            score -= 0.35
            reasons.append("small_area_penalty")
        if height_ratio > 0.55 and width_ratio < 0.45:
            score -= 0.25
            reasons.append("vertical_area_penalty")
    else:
        bed_sofa_couch_target = is_bed_sofa_couch_target(object_name)
        if area_ratio > 0.22:
            if bed_sofa_couch_target:
                score -= 0.35 * area_ratio
                reasons.append("bed_sofa_large_entity_softened")
            else:
                score -= 1.5 * area_ratio
                reasons.append("large_entity_penalty")
        if width_ratio > 0.65 or height_ratio > 0.70:
            if bed_sofa_couch_target:
                score -= 0.10
                reasons.append("bed_sofa_broad_entity_softened")
            else:
                score -= 0.35
                reasons.append("broad_entity_penalty")
        if bed_sofa_couch_target:
            furniture_bonus, furniture_reasons = bed_sofa_couch_candidate_score(box, image_pil.size)
            score += furniture_bonus
            reasons.extend(furniture_reasons)
        if area_ratio < 0.002:
            score -= 0.15
            reasons.append("tiny_entity_penalty")

    for selected_name, selected_box in selected_boxes.items():
        overlap = box_iou(box, selected_box)
        if overlap > 0.55 and is_area_object(selected_name) and not area_kind:
            score -= 0.8
            reasons.append("overlaps_area_object")
        if overlap > 0.55 and area_kind and not is_area_object(selected_name):
            score -= 0.15
            reasons.append("overlaps_entity_object")
        if is_area_object(selected_name) and not area_kind:
            if box_center_inside(box, selected_box) and box_area(box) < box_area(selected_box) * 0.65:
                score += 0.2
                reasons.append("inside_area_object")

    return score, reasons

def select_candidate_with_vlm(
    vlm_model: Any,
    image_pil: Image.Image,
    object_name: str,
    candidates: List[Dict[str, object]],
    save_dir: Optional[Union[str, Path]] = None,
    question: str = "",
    selected_boxes: Optional[Dict[str, List[int]]] = None,
) -> Optional[Dict[str, object]]:
    grid = save_candidate_grid(image_pil, object_name, candidates, save_dir)
    overlay_path = save_candidate_overlay(image_pil, object_name, candidates, save_dir)
    overlay_image = Image.open(overlay_path).convert("RGB") if overlay_path is not None else make_candidate_overlay(image_pil, candidates)
    object_kind = "area" if is_area_object(object_name) else "entity"
    relation_context = parse_object_relation_context(object_name)
    candidate_metadata = format_candidate_metadata(candidates, image_pil.size)
    prompt = f"""
Choose the single numbered bounding box that best matches the target object in the full image.

Target object phrase: {object_name}
Main target: {relation_context['target_phrase']}
Relation context: {relation_context['relation_context'] or 'none'}
Reference object: {relation_context['reference_object'] or 'none'}
Object kind: {object_kind}
Question context: {question}

Candidate metadata table:
{candidate_metadata}

Candidate 0 is the rule-ranked best candidate, but you may choose another candidate if it better matches the exact target object category and question context.

Selection rules:
- Select the candidate that corresponds to the object referred to in the question, not just the most visually obvious object of the category.
- Use spatial/contextual clues in the question, such as color, material, shape, relative position, nearby objects, and role in the scene.
- If the question distinguishes similar objects, choose the candidate matching the distinguishing phrase, such as "white coffee table", "circular table", "black table", "person wearing a hat", "left chair", or "closer sofa".
- For closest/furthest/nearest/farthest targets, the distance word is part of the target phrase. Do not choose the largest, clearest, or highest-score candidate if it violates closest/furthest from the camera/image viewpoint.
- Example: Target "furthest leather chair" means select the leather chair farthest from the camera/image viewpoint, not the large foreground chair.
- In most Omni3D-Bench questions, different relevant objects should still have some visible difference in category, size, position, extent, or role. If two candidates for different object names have very similar boxes, sizes, and positions, re-read the question carefully: you may be confusing two similarly named but distinct objects, or selecting the same physical object for both targets.
- Do not assume highly overlapping or near-identical candidates are interchangeable. Use subtle differences in box extent, center, object boundary, support surface, and scene role to distinguish related objects such as TV vs TV stand, cabinet vs cabinet top, table vs tabletop, chair vs cushion, or bed vs bedding.
- For paired or adjacent objects such as TV and TV stand, the boxes may be close and partially overlapping, but they are not the same target: TV usually refers to the screen/display, while TV stand refers to the supporting furniture below or around it.
- Match the Main target first. Use the Relation context only to choose among candidates for the same target category.
- Relation context such as "under the tv" means choose the target object located under the TV; it does not mean choose the TV.
- Relation context such as "right of X", "left of X", "next to X", "in front of X", and "behind X" is a candidate position constraint.
- Use the full-image overlay first to understand the object's position in the whole scene.
- Use the crop grid only as supporting evidence for object identity and box quality.
- Prefer complete visible objects over partial parts when the target is a complete object.
- Avoid boxes that include multiple unrelated objects or large background regions unless the target is an area object such as carpet, rug, floor, wall, or ceiling.

Object-specific rules:
- For entity objects such as white coffee table, black table, cabinet, chair, sofa, bed, plant, person, or car, avoid boxes that include large carpet/floor/background regions or multiple objects.
- For complete objects such as cabinet, table, sofa, bed, or chair, prefer the full visible object and avoid partial boxes that only cover the top, seat, backrest, leg, or one component.
- For bed, sofa, and couch targets, prefer the complete visible main body. A reasonably larger and more central complete box is usually better than an edge partial box.
- For bed, sofa, and couch targets, do not choose boxes that only cover an armrest, cushion, seat patch, bed corner, headboard fragment, or edge-truncated part.
- For bed, sofa, and couch targets, explicit position words such as leftmost, rightmost, topmost, bottommost, under, next to, or right of override the center preference.
- For carpet/rug/floor, prefer the large low horizontal floor-covering region and do not choose fireplace platforms, tabletops, beds, sofas, or other flat surfaces.
- For entity objects, avoid overly large boxes that mostly cover floor/carpet/background.

Similar-object rules:
- For visually similar but different categories such as chair, stool, bench, sofa, couch, ottoman, and seat, choose the candidate that matches the exact target category in the question.
- Do not select a stool for "chair" if a chair candidate is available.
- Do not select a chair for "stool" if a stool candidate is available.
- Do not select a chair for "bench", "sofa", "couch", or "ottoman" unless the target object in the question is actually "chair".
- Do not select an ottoman for "stool" or a stool for "ottoman" unless the question uses a generic phrase such as "seat" and no exact candidate exists.
- Use visual cues to distinguish them:
  - chair: usually has a backrest and may have arms;
  - stool: usually has no backrest and is smaller/taller as a simple seat;
  - bench: usually elongated and can seat multiple people;
  - sofa/couch: usually larger, padded, and designed for multiple people;
  - ottoman: usually a low padded seat or footrest, often without a backrest.
- If the question contains an attribute such as color, material, size, or relative location, prefer the candidate matching both the category and the attribute.

Material, shape, and color rules:
- Treat material and appearance words such as glass, translucent, transparent, clear, see-through, wooden, metal, plastic, leather, fabric, marble, and ceramic as part of the target object phrase.
- Treat shape words such as circular, round, square, rectangular, and oval as part of the target object phrase.
- Treat color words such as gray, black, white, red, and brown as part of the target object phrase when they distinguish same-category objects.
- For "gray chair" vs "black chair", choose the chair matching the requested color; do not choose a different colored chair because it is larger or clearer.
- For "polka-dot", "striped", "solid", "translucent", "glass", "circular", or "round" targets, the attribute is mandatory.
- For "glass table", choose the actual glass/transparent table, not a black plastic table or ordinary dark table.
- For "translucent cube", choose the transparent/translucent cube, not an opaque box or ordinary cube-like object.
- For "circular table" or "round table", choose the round/circular table, not a rectangular table.
- Do not choose a visually different material, color, transparency, or shape just because it has a higher detector score.
- Use box2d and center metadata for left/right/top/bottom constraints, but do not let dino_score or rank_score override explicit attributes.

Omnitest2 badcase-guided rules:
- TV and TV stand are different targets. If the target is TV stand, do not select the TV screen; if the target is TV, do not select the furniture stand.
- For "two sinks", "both sinks", repeated chairs, repeated books, repeated shelves, repeated lights, or count-ratio targets, choose candidates that can represent distinct visible instances; do not collapse different instances into one aggregate box when the target asks for multiple objects.
- Do not select answer words or non-object fragments such as decimal, sum, direction, square, format, closer, furthest point, or can you fit.
- Do not select a color word by itself unless the candidate is a physical object described by that color.
- For books, letters, handles, lights, shelves, stickers, notes, placemats, grates, empty squares, and similar small count targets, prefer tight single-instance boxes over large group/container boxes.
- For counting, do not treat a box that contains multiple visible instances as one instance; reject group boxes, container boxes, and duplicate boxes.
- For post-it notes, handles, letters, and lightbulbs, prefer each single visible object with a tight box and reject large regions that contain several targets.

Relation modifier rules:
- Treat words such as rightmost, leftmost, topmost, bottommost, center, and middle as part of the target object phrase, not as optional context.
- For "rightmost chair" or similar targets, choose the rightmost complete candidate among candidates that match the target category.
- For "center cabinet" or "middle cabinet" targets, choose the complete same-category candidate whose center is closest to the image center; do not choose a clearer or higher-scoring left/right candidate.
- Do not choose a larger, clearer, or more central candidate if it violates the explicit rightmost/leftmost/topmost/bottommost/center/middle modifier.

Return format:
- Return only one integer index.
- If none of the numbered candidates match the target object, return INVALID.
"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": overlay_image},
                {"type": "image", "image": grid},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    response = vlm_model.process_messages(messages, max_new_tokens=32)
    selected_idx = parse_candidate_index_or_none(response, len(candidates))
    if selected_idx is None:
        return None
    selected = dict(candidates[selected_idx])
    selected["candidate_index"] = selected_idx
    selected["vlm_response"] = str(response)
    if overlay_path is not None:
        selected["candidate_overlay_path"] = str(overlay_path)
    return selected


def vlm_selection_is_plausible(object_name: str, selected: Dict[str, object], candidates: List[Dict[str, object]]) -> bool:
    return validate_vlm_selection(object_name, selected, candidates)[0]


LARGE_OBJECT_TERMS = {
    "sofa", "couch", "bed", "fireplace", "cabinet", "bookshelf", "shelf",
    "chandelier", "table", "dresser", "counter", "nightstand", "stand",
}


def _is_large_object_target(object_name: str) -> bool:
    target = parse_object_relation_context(object_name).get("target_phrase", object_name).lower()
    if is_area_object(target):
        return False
    words = set(re.findall(r"[a-z]+", target))
    return any(term in words or term + "s" in words for term in LARGE_OBJECT_TERMS)


BED_SOFA_COUCH_TERMS = {"bed", "sofa", "couch"}


def is_bed_sofa_couch_target(object_name: str) -> bool:
    target = parse_object_relation_context(object_name).get("target_phrase", object_name).lower()
    if is_area_object(target):
        return False
    words = set(re.findall(r"[a-z]+", target))
    return any(term in words or term + "s" in words for term in BED_SOFA_COUCH_TERMS)


def bed_sofa_couch_candidate_score(box: List[int], image_size: Tuple[int, int]) -> Tuple[float, List[str]]:
    width, height = image_size
    image_area = float(max(1, width * height))
    area_ratio = box_area(box) / image_area
    box_w = max(0, int(box[2]) - int(box[0]))
    box_h = max(0, int(box[3]) - int(box[1]))
    width_ratio = box_w / float(max(1, width))
    height_ratio = box_h / float(max(1, height))
    cx, cy = box_center(box)
    center_dx = abs(cx - width / 2.0) / float(max(1.0, width / 2.0))
    center_dy = abs(cy - height / 2.0) / float(max(1.0, height / 2.0))
    centrality = max(0.0, 1.0 - (center_dx + center_dy) / 2.0)

    score = 0.0
    reasons: List[str] = []
    if 0.06 <= area_ratio <= 0.65:
        score += min(0.28, area_ratio * 0.70)
        reasons.append("bed_sofa_area_bonus")
    if centrality >= 0.45:
        score += 0.18 * centrality
        reasons.append("bed_sofa_center_bonus")
    if width_ratio >= 0.30 and height_ratio >= 0.20 and area_ratio >= 0.08:
        score += 0.18
        reasons.append("bed_sofa_complete_bonus")
    if area_ratio < 0.05:
        score -= 0.25
        reasons.append("bed_sofa_partial_penalty")
    if cx < width * 0.12 or cx > width * 0.88 or cy < height * 0.10 or cy > height * 0.90:
        score -= 0.20
        reasons.append("bed_sofa_edge_penalty")
    touches_edge = int(box[0]) <= 1 or int(box[1]) <= 1 or int(box[2]) >= width - 1 or int(box[3]) >= height - 1
    if touches_edge and area_ratio < 0.35:
        score -= 0.15
        reasons.append("bed_sofa_edge_penalty")
    return score, reasons


def _is_nearly_whole_image_candidate(candidate: Dict[str, object]) -> bool:
    box = candidate.get("box2d", [0, 0, 0, 0])
    area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
    if area_ratio <= 0.0:
        # Candidate metadata may not include image-relative area in unit tests.
        return False
    return area_ratio > 0.75


def _large_object_complete_box_override(object_name: str, selected: Dict[str, object], best: Dict[str, object]) -> bool:
    if not is_bed_sofa_couch_target(object_name):
        return False
    if not _candidate_matches_target_phrase(object_name, selected):
        return False
    if _is_nearly_whole_image_candidate(selected):
        return False
    selected_area = box_area(selected.get("box2d", [0, 0, 0, 0]))
    best_area = max(1, box_area(best.get("box2d", [0, 0, 0, 0])))
    return selected_area <= best_area * 8.0


def validate_vlm_selection(object_name: str, selected: Dict[str, object], candidates: List[Dict[str, object]], image_size: Optional[Tuple[int, int]] = None) -> Tuple[bool, Optional[str]]:
    if not candidates:
        return False, "no_candidates"
    selected_score = float(selected.get("rank_score", selected.get("score", 0.0)))
    best_score = float(candidates[0].get("rank_score", candidates[0].get("score", 0.0)))
    relation = relation_modifier(object_name)
    if _candidate_matches_reference_object(object_name, selected):
        return False, "reference_object_selected"
    if _candidate_crosses_tv_stand_boundary(object_name, selected):
        return False, "tv_tv_stand_mismatch"
    if _attribute_target_mismatch(object_name, selected, candidates):
        return False, "attribute_mismatch"
    rank_margin = 0.45 if is_material_object(object_name) else 0.20
    if selected_score < best_score - rank_margin:
        if not (relation in {"closest", "furthest"} and _selection_matches_relation(object_name, selected, candidates, image_size=image_size)):
            return False, "rank_score_too_low"
    if not _selection_matches_relation(object_name, selected, candidates, image_size=image_size):
        return False, "relation_mismatch"
    if not is_area_object(object_name):
        selected_area = box_area(selected.get("box2d", [0, 0, 0, 0]))
        best_area = max(1, box_area(candidates[0].get("box2d", [0, 0, 0, 0])))
        if selected_score < best_score and selected_area < best_area * 0.70 and _prefers_complete_entity(object_name):
            if not _relation_target_candidate_override(object_name, selected):
                return False, "partial_entity_box"
        if selected_score < best_score and selected_area > best_area * 2.0:
            if not (
                (_attribute_target_candidate_override(object_name, selected) and selected_area <= best_area * 4.0)
                or _large_object_complete_box_override(object_name, selected, candidates[0])
            ):
                return False, "entity_box_too_large"
    return True, None

def save_candidate_grid(
    image_pil: Image.Image,
    object_name: str,
    candidates: List[Dict[str, object]],
    save_dir: Optional[Union[str, Path]],
) -> Image.Image:
    crops = []
    for idx, candidate in enumerate(candidates):
        box = clamp_box_to_image(candidate["box2d"], image_pil.size)
        crop = image_pil.crop(tuple(box)).resize((224, 224))
        draw = ImageDraw.Draw(crop)
        draw.rectangle([0, 0, 44, 30], fill="white")
        draw.text((8, 6), str(idx), fill="red")
        crops.append(crop)
    cols = min(5, max(1, len(crops)))
    rows = int(math.ceil(len(crops) / float(cols)))
    grid = Image.new("RGB", (cols * 224, rows * 224), "white")
    for idx, crop in enumerate(crops):
        grid.paste(crop, ((idx % cols) * 224, (idx // cols) * 224))
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = object_name.replace("/", "_").replace(" ", "_")
        grid.save(out_dir / f"detection_candidates_{safe_name}.png")
    return grid


def save_candidate_overlay(
    image_pil: Image.Image,
    object_name: str,
    candidates: List[Dict[str, object]],
    save_dir: Optional[Union[str, Path]],
) -> Optional[Path]:
    if save_dir is None:
        return None
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = object_name.replace("/", "_").replace(" ", "_")
    overlay = make_candidate_overlay(image_pil, candidates)
    output_path = out_dir / f"detection_candidates_{safe_name}_overlay.png"
    overlay.save(output_path)
    return output_path


def make_candidate_overlay(image_pil: Image.Image, candidates: List[Dict[str, object]]) -> Image.Image:
    overlay = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange", "white", "blue"]
    for idx, candidate in enumerate(candidates):
        box = clamp_box_to_image(candidate["box2d"], image_pil.size)
        color = colors[idx % len(colors)]
        draw_labeled_box(draw, box, color, width=4)
        label = str(idx)
        x1, y1 = box[0], box[1]
        draw.rectangle([x1, y1, x1 + 34, y1 + 28], fill="black")
        draw.text((x1 + 8, y1 + 5), label, fill=color)
    return overlay



def draw_labeled_box(draw: ImageDraw.ImageDraw, box: List[int], color: str, width: int = 4) -> None:
    for offset in range(width):
        draw.rectangle(
            [box[0] + offset, box[1] + offset, box[2] - offset, box[3] - offset],
            outline=color,
        )

def maybe_fallback_mask(
    image_pil: Image.Image,
    object_name: str,
    box2d: List[int],
    mask,
) -> Tuple[Any, Optional[str]]:
    if is_area_object(object_name):
        return mask, None

    width, height = image_pil.size
    box = clamp_box_to_image(box2d, image_pil.size)
    area_ratio = box_area(box) / float(max(1, width * height))
    box_w = max(0, box[2] - box[0])
    box_h = max(0, box[3] - box[1])
    width_ratio = box_w / float(max(1, width))
    height_ratio = box_h / float(max(1, height))
    mask_area = int((mask > 0.5).sum())
    box_area_value = max(1, box_area(box))
    mask_box_ratio = mask_area / float(box_area_value)

    reason = None
    if area_ratio > 0.35 or width_ratio > 0.78 or height_ratio > 0.78:
        reason = "entity_box_too_large"
    elif mask_box_ratio > 0.9 and area_ratio > 0.30:
        reason = "entity_mask_too_broad"

    if reason is None:
        return mask, None
    return central_box_mask(image_pil, box), reason


def central_box_mask(image_pil: Image.Image, box: List[int]):
    import numpy as np

    mask = np.zeros((image_pil.height, image_pil.width), dtype=np.float32)
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx1 = x1 + int(w * 0.25)
    cx2 = x2 - int(w * 0.25)
    cy1 = y1 + int(h * 0.25)
    cy2 = y2 - int(h * 0.25)
    if cx1 >= cx2 or cy1 >= cy2:
        mask[y1:y2, x1:x2] = 1.0
    else:
        mask[cy1:cy2, cx1:cx2] = 1.0
    return mask


def overlap_warnings(object_name: str, box2d: List[int], selected_boxes: Dict[str, List[int]]) -> List[Dict[str, object]]:
    warnings = []
    for selected_name, selected_box in selected_boxes.items():
        overlap = box_iou(box2d, selected_box)
        if selected_name == object_name:
            continue
        if overlap > 0.55 and (is_area_object(object_name) or is_area_object(selected_name)):
            warnings.append({"object": selected_name, "iou": float(overlap)})
        elif overlap > 0.85 and not is_area_object(object_name) and not is_area_object(selected_name):
            warnings.append({
                "object": selected_name,
                "iou": float(overlap),
                "warning": "different_non_area_high_overlap",
            })
    return warnings


def summarize_candidates(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary = []
    for candidate in candidates:
        item = {
            "prompt": candidate.get("prompt", ""),
            "box2d": [int(v) for v in candidate.get("box2d", [])],
        }
        if "score" in candidate:
            item["score"] = float(candidate["score"])
        if "phrase" in candidate:
            item["phrase"] = candidate["phrase"]
        if "candidate_index" in candidate:
            item["candidate_index"] = int(candidate["candidate_index"])
        if "rank_score" in candidate:
            item["rank_score"] = float(candidate["rank_score"])
        if "rank_reasons" in candidate:
            item["rank_reasons"] = list(candidate["rank_reasons"])
        summary.append(item)
    return summary


def format_candidate_metadata(candidates: List[Dict[str, object]], image_size: Tuple[int, int]) -> str:
    width, height = image_size
    lines = ["index | prompt | box2d | center | area_ratio | dino_score | rank_score | rank_reasons"]
    image_area = float(max(1, width * height))
    for idx, candidate in enumerate(candidates):
        box = candidate.get("box2d", [0, 0, 0, 0])
        cx, cy = box_center(box)
        area_ratio = box_area(box) / image_area
        dino_score = float(candidate.get("score", 0.0))
        rank_score = float(candidate.get("rank_score", dino_score))
        reasons = ",".join(str(reason) for reason in candidate.get("rank_reasons", [])) or "none"
        lines.append(
            f"{idx} | {candidate.get('prompt', '')} | {[int(v) for v in box]} | "
            f"[{cx:.1f}, {cy:.1f}] | {area_ratio:.4f} | {dino_score:.3f} | {rank_score:.3f} | {reasons}"
        )
    lines.append("Coordinate hint: larger x means farther right in the image; larger y means lower in the image. Scores are supporting evidence, not a replacement for target attributes and relation context.")
    return "\n".join(lines)


def parse_candidate_index(response: object, num_candidates: int) -> int:
    selected = parse_candidate_index_or_none(response, num_candidates)
    return 0 if selected is None else selected


def parse_candidate_index_or_none(response: object, num_candidates: int) -> Optional[int]:
    match = re.search(r"\d+", str(response))
    if not match:
        return None
    idx = int(match.group(0))
    if idx < 0 or idx >= num_candidates:
        return None
    return idx


def parse_object_relation_context(object_name: str) -> Dict[str, str]:
    text = re.sub(r"\s+", " ", str(object_name or "").strip().lower())
    patterns = [
        (r"\s+(closest to|nearest to)\s+", "closest to"),
        (r"\s+(furthest from|farthest from|furthest to|farthest to)\s+", "furthest from"),
        (r"\s+(to the right of|right of)\s+", "right of"),
        (r"\s+(to the left of|left of)\s+", "left of"),
        (r"\s+(in front of|front of)\s+", "in front of"),
        (r"\s+(next to|beside|near)\s+", "next to"),
        (r"\s+(underneath|under|below|beneath)\s+", "under"),
        (r"\s+(above|over|on top of)\s+", "above"),
        (r"\s+(behind)\s+", "behind"),
    ]
    for pattern, relation in patterns:
        match = re.search(pattern, text)
        if match:
            target = text[:match.start()].strip()
            reference = text[match.end():].strip()
            reference = re.sub(r"^(?:the|a|an)\s+", "", reference).strip()
            return {
                "target_phrase": target or text,
                "relation": relation,
                "relation_context": text[match.start():].strip(),
                "reference_object": reference,
            }
    return {"target_phrase": text, "relation": "", "relation_context": "", "reference_object": ""}


def _remove_count_quantifier(prompt: str) -> str:
    return re.sub(
        r"^(?:the\s+)?(?:two|both|multiple|all)\s+",
        "",
        str(prompt or "").strip(),
    ).strip()


def _remove_relation_words(prompt: str) -> str:
    prompt = prompt.replace("left-most", "leftmost").replace("top-most", "topmost")
    prompt = _remove_count_quantifier(prompt)
    return re.sub(
        r"^(?:rightmost|leftmost|center|topmost|bottommost|combined|same|double)\s+",
        "",
        prompt,
    ).strip()


def _remove_leading_color(prompt: str) -> str:
    return re.sub(
        r"^(?:white|black|red|blue|green|yellow|brown|gray|grey|pink|purple|orange|silver|gold|wooden|dark|light)\s+",
        "",
        prompt,
    ).strip()


SAME_TYPE_EXISTENCE_RE = re.compile(
    r"\b(?:two|2|multiple|more than one|same)\b.*\b(?:same object types?|same objects?|object types?)\b|"
    r"\b(?:are|is) there (?:two|2|multiple|more than one) of the same objects?\b",
    re.IGNORECASE,
)
NON_VISUAL_COUNT_RE = re.compile(r"\b(?:need|needed|stack|stacked|achieve|match|reach|same height|have to)\b")
COUNT_RATIO_RE = re.compile(r"\bratio\s+of\b.*\b(?:to|and)\b", re.IGNORECASE)
COUNT_COMPARISON_RE = re.compile(r"\b(?:more|fewer|less|greater|larger|smaller|most|fewest|least)\b", re.IGNORECASE)
NUMERIC_DIMENSION_RE = re.compile(r"\b(?:height|width|length|depth|volume|distance|size|area|diagonal)\b", re.IGNORECASE)
MULTI_INSTANCE_NUMERIC_PATTERN = (
    r"\b(?:two|both|multiple|all)\s+(?:of\s+)?(?:the\s+)?{object}\b|"
    r"\bcombined\s+(?:height|width|length|depth|volume|size|area)\s+of\s+(?:the\s+)?(?:two|both|multiple|all)\s+(?:of\s+)?(?:the\s+)?{object}\b"
)


def is_same_type_existence_question(question: str) -> bool:
    return bool(SAME_TYPE_EXISTENCE_RE.search(str(question or "")))


def is_count_ratio_question(question: str) -> bool:
    question_text = re.sub(r"\s+", " ", str(question or "").lower())
    return bool(COUNT_RATIO_RE.search(question_text)) and not bool(NUMERIC_DIMENSION_RE.search(question_text))


def is_same_category_multi_instance_numeric(question: str, object_name: str) -> bool:
    question_text = re.sub(r"\s+", " ", str(question or "").lower())
    forms = _object_name_forms(_remove_count_quantifier(object_name))
    for form in forms:
        if not form:
            continue
        pattern = MULTI_INSTANCE_NUMERIC_PATTERN.format(object=re.escape(form))
        if re.search(pattern, question_text, re.IGNORECASE):
            return True
    return False


def _object_mentioned_in_question(question_text: str, object_name: str) -> bool:
    for form in _object_name_forms(_remove_count_quantifier(object_name)):
        if form and re.search(r"\b" + re.escape(form) + r"\b", question_text):
            return True
    return False


def is_visual_count_target(question_type: Optional[str], question: str, object_name: str) -> bool:
    question_text = re.sub(r"\s+", " ", str(question or "").lower())
    if is_same_type_existence_question(question_text):
        return True
    if is_same_category_multi_instance_numeric(question_text, object_name):
        return True
    if is_count_ratio_question(question_text) and _object_mentioned_in_question(question_text, object_name):
        return True
    if COUNT_COMPARISON_RE.search(question_text) and _object_mentioned_in_question(question_text, object_name):
        return True
    if NON_VISUAL_COUNT_RE.search(question_text):
        return False
    forms = _object_name_forms(object_name)
    for form in forms:
        escaped = re.escape(form)
        if re.search(r"\bhow many\s+" + escaped + r"\b", question_text):
            return True
        if re.search(r"\bnumber of\s+" + escaped + r"\b", question_text):
            return True
    return False


def _object_name_forms(object_name: str) -> List[str]:
    name = re.sub(r"\s+", " ", str(object_name or "").strip().lower())
    singular = singularize_count_name(name)
    plural = pluralize_count_name(singular)
    return _dedupe_preserve_order([name, singular, plural])


def singularize_count_name(name: str) -> str:
    value = re.sub(r"\s+", " ", str(name or "").strip().lower())
    if value.endswith("ies") and len(value) > 3:
        return value[:-3] + "y"
    if value.endswith("ves") and len(value) > 3:
        return value[:-3] + "f"
    if value.endswith("xes") or value.endswith("ches") or value.endswith("shes") or value.endswith("ses"):
        return value[:-2]
    if value.endswith("s") and not value.endswith("ss") and len(value) > 1:
        return value[:-1]
    return value


def pluralize_count_name(name: str) -> str:
    value = re.sub(r"\s+", " ", str(name or "").strip().lower())
    if value.endswith("y"):
        return value[:-1] + "ies"
    if value.endswith(("x", "ch", "sh", "s")):
        return value + "es"
    return value + "s"


def make_count_instance_name(object_name: str, idx: int) -> str:
    prefix = singularize_count_name(_remove_leading_color(_remove_relation_words(object_name.lower())))
    prefix = re.sub(r"[^a-z0-9]+", "_", prefix).strip("_") or "object"
    return f"{prefix}_{idx}"


def select_count_instance_candidates(candidates: List[Dict[str, object]], max_instances: int = 20) -> List[Dict[str, object]]:
    selected, _ = filter_count_instance_candidates(candidates, max_instances=max_instances)
    return selected


def filter_count_instance_candidates(
    candidates: List[Dict[str, object]],
    max_instances: int = 20,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not candidates:
        return [], {"summary": {"input_count": 0, "selected_count": 0, "rejected_count": 0}, "rejected_candidates": []}
    best_score = float(candidates[0].get("rank_score", candidates[0].get("score", 0.0)))
    score_floor = best_score - 0.35
    eligible = [
        candidate for candidate in candidates
        if float(candidate.get("rank_score", candidate.get("score", 0.0))) >= score_floor
    ]
    rejected: List[Dict[str, object]] = []

    def reject(candidate: Dict[str, object], reason: str, other: Optional[Dict[str, object]] = None) -> None:
        item = {
            "candidate_index": candidate.get("candidate_index"),
            "box2d": candidate.get("box2d"),
            "reject_reason": reason,
        }
        if other is not None:
            item["overlap_with"] = other.get("candidate_index")
            item["contained_by"] = other.get("candidate_index") if reason in {"contained_by_tighter_box", "group_box_contains_instances"} else None
        rejected.append(item)

    filtered = []
    for candidate in eligible:
        box = candidate.get("box2d", [0, 0, 0, 0])
        if _candidate_contains_multiple_tighter_boxes(candidate, eligible):
            reject(candidate, "group_box_contains_instances")
            continue
        duplicate_of = _find_count_duplicate(candidate, filtered, image_size)
        if duplicate_of is not None:
            reject(candidate, "duplicate_or_center_close", duplicate_of)
            continue
        containing = _find_tighter_contained_candidate(candidate, eligible)
        if containing is not None:
            reject(candidate, "contained_by_tighter_box", containing)
            continue
        filtered.append(candidate)

    selected = filtered[: max(1, max_instances)]
    if len(filtered) > max_instances:
        for candidate in filtered[max_instances:]:
            reject(candidate, "max_instances_limit")
    selected.sort(key=lambda item: (box_center(item.get("box2d", [0, 0, 0, 0]))[1], box_center(item.get("box2d", [0, 0, 0, 0]))[0]))
    return selected, {
        "summary": {
            "input_count": len(candidates),
            "eligible_count": len(eligible),
            "selected_count": len(selected),
            "rejected_count": len(rejected),
        },
        "rejected_candidates": rejected,
    }


def _find_count_duplicate(candidate: Dict[str, object], selected: List[Dict[str, object]], image_size: Optional[Tuple[int, int]]) -> Optional[Dict[str, object]]:
    box = candidate.get("box2d", [0, 0, 0, 0])
    for item in selected:
        other_box = item.get("box2d", [0, 0, 0, 0])
        if box_iou(box, other_box) > 0.65:
            return item
        if image_size is not None and box_center_distance_ratio(box, other_box, image_size) < 0.035:
            area_a = max(1, box_area(box))
            area_b = max(1, box_area(other_box))
            if 0.45 <= area_a / float(area_b) <= 2.2:
                return item
    return None


def _find_tighter_contained_candidate(candidate: Dict[str, object], candidates: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    box = candidate.get("box2d", [0, 0, 0, 0])
    area_value = max(1, box_area(box))
    for other in candidates:
        if other is candidate:
            continue
        other_box = other.get("box2d", [0, 0, 0, 0])
        other_area = max(1, box_area(other_box))
        if other_area >= area_value * 0.85:
            continue
        if box_containment_ratio(other_box, box) >= 0.90:
            return other
    return None


def _candidate_contains_multiple_tighter_boxes(candidate: Dict[str, object], candidates: List[Dict[str, object]]) -> bool:
    box = candidate.get("box2d", [0, 0, 0, 0])
    area_value = max(1, box_area(box))
    contained = 0
    for other in candidates:
        if other is candidate:
            continue
        other_box = other.get("box2d", [0, 0, 0, 0])
        other_area = max(1, box_area(other_box))
        if other_area >= area_value * 0.65:
            continue
        if box_containment_ratio(other_box, box) >= 0.88:
            contained += 1
            if contained >= 2:
                return True
    return False


def _camera_distance_image_proxy(candidate: Dict[str, object], image_size: Tuple[int, int]) -> float:
    box = candidate.get("box2d", [0, 0, 0, 0])
    width, height = image_size
    area_ratio = box_area(box) / float(max(1, width * height))
    box_h = max(0.0, float(box[3]) - float(box[1])) / float(max(1, height))
    _, cy = box_center(box)
    bottom_ratio = float(box[3]) / float(max(1, height))
    center_y_ratio = cy / float(max(1, height))
    return area_ratio * 0.55 + box_h * 0.20 + bottom_ratio * 0.15 + center_y_ratio * 0.10


def relation_modifier(object_name: str) -> str:
    name = object_name.lower()
    name = name.replace("left-most", "leftmost").replace("right-most", "rightmost")
    name = name.replace("top-most", "topmost").replace("bottom-most", "bottommost")
    for modifier in ["leftmost", "rightmost", "center", "centered", "middle", "topmost", "bottommost", "closest", "nearest", "furthest", "farthest"]:
        if name.startswith(modifier + " "):
            if modifier in {"centered", "middle"}:
                return "center"
            if modifier == "nearest":
                return "closest"
            if modifier == "farthest":
                return "furthest"
            return modifier
    return ""


def select_by_relation(ranked, relation: str, width: int, height: int) -> Dict[str, object]:
    raw_candidates = [item[2] for item in ranked]
    candidates = raw_candidates if relation in {"closest", "furthest"} else _reasonable_relation_candidates(raw_candidates, relation)
    if relation == "leftmost":
        return min(candidates, key=lambda item: box_center(item["box2d"])[0])
    if relation == "rightmost":
        return max(candidates, key=lambda item: box_center(item["box2d"])[0])
    if relation == "topmost":
        return min(candidates, key=lambda item: box_center(item["box2d"])[1])
    if relation == "bottommost":
        return max(candidates, key=lambda item: box_center(item["box2d"])[1])
    if relation == "center":
        image_center = (width / 2.0, height / 2.0)
        return min(candidates, key=lambda item: distance(box_center(item["box2d"]), image_center))
    if relation == "closest":
        return max(candidates, key=lambda item: _camera_distance_image_proxy(item, (width, height)))
    if relation == "furthest":
        return min(candidates, key=lambda item: _camera_distance_image_proxy(item, (width, height)))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][2]


def _reasonable_relation_candidates(candidates: List[Dict[str, object]], relation: str = "", score_margin: float = 0.25) -> List[Dict[str, object]]:
    if not candidates:
        return []
    best_score = max(float(item.get("rank_score", item.get("score", 0.0))) for item in candidates)
    reasonable = [
        item for item in candidates
        if float(item.get("rank_score", item.get("score", 0.0))) >= best_score - score_margin
    ]
    protected = _relation_extreme_candidate(candidates, relation)
    if protected is not None and _should_protect_relation_extreme(protected):
        reasonable.append(protected)
    return _dedupe_candidates_by_box(reasonable) or candidates


def _relation_extreme_candidate(candidates: List[Dict[str, object]], relation: str) -> Optional[Dict[str, object]]:
    if not candidates:
        return None
    if relation == "leftmost":
        return min(candidates, key=lambda item: box_center(item["box2d"])[0])
    if relation == "rightmost":
        return max(candidates, key=lambda item: box_center(item["box2d"])[0])
    if relation == "topmost":
        return min(candidates, key=lambda item: box_center(item["box2d"])[1])
    if relation == "bottommost":
        return max(candidates, key=lambda item: box_center(item["box2d"])[1])
    return None


def _should_protect_relation_extreme(candidate: Dict[str, object]) -> bool:
    reasons = set(candidate.get("rank_reasons") or [])
    return bool(reasons.intersection({"large_entity_penalty", "broad_entity_penalty"}))


def _dedupe_candidates_by_box(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    result = []
    for candidate in candidates:
        box = candidate.get("box2d", [0, 0, 0, 0])
        if not any(_same_box(box, item.get("box2d", [0, 0, 0, 0])) for item in result):
            result.append(candidate)
    return result


def _same_box(a, b) -> bool:
    if a is None or b is None:
        return False
    return [int(v) for v in a] == [int(v) for v in b]


def _selection_matches_relation(object_name: str, selected: Dict[str, object], candidates: List[Dict[str, object]], image_size: Optional[Tuple[int, int]] = None) -> bool:
    relation = relation_modifier(object_name)
    if relation not in {"leftmost", "rightmost", "topmost", "bottommost", "center", "closest", "furthest"}:
        return True
    reasonable = candidates if relation in {"closest", "furthest"} else _reasonable_relation_candidates(candidates, relation)
    selected_box = selected.get("box2d", [0, 0, 0, 0])
    if not any(_same_box(selected_box, item.get("box2d", [0, 0, 0, 0])) for item in reasonable):
        return False
    if relation == "leftmost":
        expected = min(reasonable, key=lambda item: box_center(item["box2d"])[0])
        return box_center(selected_box)[0] <= box_center(expected["box2d"])[0] + 1.0
    if relation == "rightmost":
        expected = max(reasonable, key=lambda item: box_center(item["box2d"])[0])
        return box_center(selected_box)[0] >= box_center(expected["box2d"])[0] - 1.0
    if relation == "topmost":
        expected = min(reasonable, key=lambda item: box_center(item["box2d"])[1])
        return box_center(selected_box)[1] <= box_center(expected["box2d"])[1] + 1.0
    if relation == "bottommost":
        expected = max(reasonable, key=lambda item: box_center(item["box2d"])[1])
        return box_center(selected_box)[1] >= box_center(expected["box2d"])[1] - 1.0
    if relation == "center":
        image_center = (image_size[0] / 2.0, image_size[1] / 2.0) if image_size is not None else _candidate_set_center(candidates)
        expected = min(reasonable, key=lambda item: distance(box_center(item["box2d"]), image_center))
        selected_distance = distance(box_center(selected_box), image_center)
        expected_distance = distance(box_center(expected["box2d"]), image_center)
        return selected_distance <= expected_distance + 1.0
    if relation in {"closest", "furthest"}:
        size = image_size or _candidate_set_image_size(candidates)
        selected_proxy = _camera_distance_image_proxy(selected, size)
        proxies = [_camera_distance_image_proxy(item, size) for item in reasonable]
        expected_proxy = max(proxies) if relation == "closest" else min(proxies)
        tolerance = 0.03
        if relation == "closest":
            return selected_proxy >= expected_proxy - tolerance
        return selected_proxy <= expected_proxy + tolerance
    return True


def _candidate_set_image_size(candidates: List[Dict[str, object]]) -> Tuple[int, int]:
    max_x = 1
    max_y = 1
    for item in candidates:
        box = item.get("box2d", [0, 0, 0, 0])
        if box:
            max_x = max(max_x, int(box[2]))
            max_y = max(max_y, int(box[3]))
    return (max_x, max_y)

def _candidate_set_center(candidates: List[Dict[str, object]]) -> Tuple[float, float]:
    boxes = [item.get("box2d", [0, 0, 0, 0]) for item in candidates if item.get("box2d")]
    if not boxes:
        return (0.0, 0.0)
    min_x = min(float(box[0]) for box in boxes)
    min_y = min(float(box[1]) for box in boxes)
    max_x = max(float(box[2]) for box in boxes)
    max_y = max(float(box[3]) for box in boxes)
    return ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)


def is_area_object(object_name: str) -> bool:
    base = _remove_leading_color(_remove_relation_words(object_name.lower()))
    return base in {"carpet", "rug", "floor", "wall", "ceiling"}


def _prefers_complete_entity(object_name: str) -> bool:
    base = _remove_leading_color(_remove_relation_words(object_name.lower()))
    return any(word in base for word in ("cabinet", "table", "sofa", "chair"))


def is_shape_object(object_name: str) -> bool:
    return bool(re.search(r"\b(?:circular|round|square|rectangular|oval)\b", object_name.lower()))


def prompt_has_shape(prompt: str, object_name: str) -> bool:
    shapes = re.findall(r"\b(?:circular|round|square|rectangular|oval)\b", object_name.lower())
    prompt_text = prompt.lower()
    if "circular" in shapes:
        shapes.append("round")
    if "round" in shapes:
        shapes.append("circular")
    return any(shape in prompt_text for shape in shapes)


def _relation_target_candidate_override(object_name: str, selected: Dict[str, object]) -> bool:
    context = parse_object_relation_context(object_name)
    if not context.get("relation_context"):
        return False
    return _candidate_matches_target_phrase(object_name, selected)


def _attribute_target_candidate_override(object_name: str, selected: Dict[str, object]) -> bool:
    target = parse_object_relation_context(object_name).get("target_phrase", "")
    if not (is_color_object(target) or is_material_object(target) or is_shape_object(target)):
        return False
    return _candidate_matches_target_phrase(object_name, selected)


def _candidate_matches_target_phrase(object_name: str, candidate: Dict[str, object]) -> bool:
    context = parse_object_relation_context(object_name)
    target = context.get("target_phrase", "")
    prompt = str(candidate.get("prompt", "")).lower()
    if not target or not prompt:
        return False
    if prompt == target or prompt == object_name.lower():
        return True
    if prompt in detection_prompts_for_object(target):
        return True
    return False


def _candidate_crosses_tv_stand_boundary(object_name: str, candidate: Dict[str, object]) -> bool:
    target = parse_object_relation_context(object_name).get("target_phrase", object_name).lower()
    prompt = str(candidate.get("prompt", "")).lower().strip()
    if "tv stand" in target or "television stand" in target:
        return prompt in {"tv", "television", "monitor"}
    if target in {"tv", "television", "monitor"}:
        return "stand" in prompt
    return False


def _attribute_target_mismatch(object_name: str, selected: Dict[str, object], candidates: List[Dict[str, object]]) -> bool:
    target = parse_object_relation_context(object_name).get("target_phrase", object_name)
    if not (is_color_object(target) or is_material_object(target) or is_shape_object(target) or _has_pattern_attribute(target)):
        return False
    if _candidate_matches_target_phrase(target, selected):
        return False
    return any(_candidate_matches_target_phrase(target, candidate) for candidate in candidates)


def _has_pattern_attribute(object_name: str) -> bool:
    return bool(re.search(r"\b(?:polka[- ]dot|striped|solid|patterned)\b", object_name.lower()))


def _candidate_matches_reference_object(object_name: str, candidate: Dict[str, object]) -> bool:
    context = parse_object_relation_context(object_name)
    reference = context.get("reference_object", "")
    if not reference:
        return False
    prompt = str(candidate.get("prompt", "")).lower().strip()
    if not prompt:
        return False
    reference_prompts = set(detection_prompts_for_object(reference))
    if reference in {"tv", "television"}:
        reference_prompts.update({"tv", "television"})
    return prompt in reference_prompts


def is_color_object(object_name: str) -> bool:
    return bool(re.match(r"^(?:white|black|red|blue|green|yellow|brown|gray|grey|pink|purple|orange|silver|gold|dark|light)\s+", object_name.lower()))


def prompt_has_color(prompt: str, object_name: str) -> bool:
    color = object_name.lower().split()[0]
    return prompt.lower().startswith(color + " ")


def is_transparency_object(object_name: str) -> bool:
    return bool(re.search(r"\b(?:translucent|transparent|clear|see[- ]through)\b", object_name.lower()))


def _remove_leading_transparency(prompt: str) -> str:
    return re.sub(r"^(?:translucent|transparent|clear|see[- ]through)\s+", "", prompt.lower()).strip()


def _transparency_synonym_prompts(object_name: str) -> List[str]:
    base = _remove_leading_transparency(object_name)
    if not base or base == object_name.lower().strip():
        return []
    return [f"translucent {base}", f"transparent {base}", f"clear {base}", f"see-through {base}"]


def is_material_object(object_name: str) -> bool:
    return bool(re.search(r"\b(?:glass|translucent|transparent|clear|see[- ]through|wooden|metal|plastic|leather|fabric|marble|ceramic)\b", object_name.lower()))


def prompt_has_material(prompt: str, object_name: str) -> bool:
    materials = re.findall(r"\b(?:glass|translucent|transparent|clear|see[- ]through|wooden|metal|plastic|leather|fabric|marble|ceramic)\b", object_name.lower())
    prompt_text = prompt.lower()
    if "glass" in materials:
        materials.extend(["transparent", "clear"])
    if any(material in materials for material in ("translucent", "transparent", "clear", "see-through", "see through")):
        materials.extend(["translucent", "transparent", "clear", "see-through", "see through"])
    return any(material in prompt_text for material in materials)


def box_area(box: List[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def box_center(box: List[int]):
    return ((int(box[0]) + int(box[2])) / 2.0, (int(box[1]) + int(box[3])) / 2.0)


def box_center_inside(box: List[int], container: List[int]) -> bool:
    cx, cy = box_center(box)
    return int(container[0]) <= cx <= int(container[2]) and int(container[1]) <= cy <= int(container[3])


def box_iou(a: List[int], b: List[int]) -> float:
    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    inter = box_area([x1, y1, x2, y2])
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def box_containment_ratio(inner: List[int], outer: List[int]) -> float:
    inner_area = box_area(inner)
    if inner_area <= 0:
        return 0.0
    x1 = max(int(inner[0]), int(outer[0]))
    y1 = max(int(inner[1]), int(outer[1]))
    x2 = min(int(inner[2]), int(outer[2]))
    y2 = min(int(inner[3]), int(outer[3]))
    return box_area([x1, y1, x2, y2]) / float(inner_area)


def box_center_distance_ratio(a: List[int], b: List[int], image_size: Tuple[int, int]) -> float:
    width, height = image_size
    diag = math.sqrt(max(1, width) ** 2 + max(1, height) ** 2)
    return distance(box_center(a), box_center(b)) / float(max(1.0, diag))


def distance(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def clamp_box_to_image(box: List[int], size) -> List[int]:
    width, height = size
    return [
        max(0, min(width - 1, int(box[0]))),
        max(0, min(height - 1, int(box[1]))),
        max(0, min(width, int(box[2]))),
        max(0, min(height, int(box[3]))),
    ]


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _coerce_config(
    config: Optional[Union[Object3DExtractionConfig, Dict[str, Any]]]
) -> Object3DExtractionConfig:
    if config is None:
        return Object3DExtractionConfig()
    if isinstance(config, Object3DExtractionConfig):
        return config
    if isinstance(config, dict):
        cfg = Object3DExtractionConfig()
        for section_name, section_value in config.items():
            if not hasattr(cfg, section_name):
                raise ValueError(f"Unknown config section: {section_name}")
            target = getattr(cfg, section_name)
            if isinstance(section_value, dict):
                for key, value in section_value.items():
                    if not hasattr(target, key):
                        raise ValueError(f"Unknown config key: {section_name}.{key}")
                    setattr(target, key, value)
            else:
                setattr(cfg, section_name, section_value)
        return cfg
    raise TypeError("config must be None, Object3DExtractionConfig, or dict")
