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
    ) -> Dict[str, Dict[str, object]]:
        image_pil = load_rgb_image(image)
        names = validate_object_names(object_names)

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
            try:
                is_count_target = is_visual_count_target(question_type, question, object_name)
                max_candidates = None
                if is_count_target:
                    max_candidates = max(20, self.config.detection.num_candidates * len(prompts_tried) * 4)
                candidates = self._collect_detection_candidates(
                    image_pil,
                    object_name,
                    prompts_tried,
                    max_candidates=max_candidates,
                )
                if not candidates:
                    results[object_name] = {
                        "error": f"No detection for object: {object_name}",
                        "prompts_tried": prompts_tried,
                    }
                    continue

                if is_count_target:
                    instance_results, instance_names, depth, counting_info = self._extract_count_instances(
                        image_pil=image_pil,
                        object_name=object_name,
                        candidates=candidates,
                        selected_boxes=selected_boxes,
                        depth=depth,
                        visualize=visualize,
                        save_dir=save_dir,
                        question=question,
                    )
                    results[object_name] = {
                        "error": "Count target expanded into indexed instances",
                        "counting_target": True,
                        "counting_instances": instance_names,
                        "counting_instance_count": len(instance_names),
                        "prompts_tried": prompts_tried,
                        "candidates_considered": summarize_candidates(candidates),
                    }
                    results[object_name].update(counting_info)
                    results.update(instance_results)
                    continue

                best_detection = self._select_detection(
                    image_pil=image_pil,
                    object_name=object_name,
                    candidates=candidates,
                    selected_boxes=selected_boxes,
                    save_dir=save_dir if visualize else None,
                    question=question,
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
                        "mask_fallback_reason": mask_fallback_reason,
                        "candidate_overlay_path": best_detection.get("candidate_overlay_path"),
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
                    "mask_fallback_reason": mask_fallback_reason,
                    "candidate_overlay_path": best_detection.get("candidate_overlay_path"),
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
        max_candidates: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        candidates = []
        for prompt in prompts:
            detections = self.detection_module.detect(image_pil, prompt)
            for det in detections:
                candidate = dict(det)
                candidate["prompt"] = prompt
                candidate["object_name"] = object_name
                candidate["box2d"] = [int(v) for v in candidate["box2d"]]
                candidates.append(candidate)
        candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        limit = max_candidates if max_candidates is not None else self.config.detection.num_candidates * len(prompts)
        candidates = candidates[: max(1, limit)]
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
        question: str = "",
    ) -> Tuple[Dict[str, Dict[str, object]], List[str], Any, Dict[str, object]]:
        instances = {}  # type: Dict[str, Dict[str, object]]
        instance_names = []  # type: List[str]
        max_instances = max(20, self.config.detection.num_candidates * 4)
        counting_info = {
            "counting_vlm_response": None,
            "counting_vlm_selected_indices": [],
            "counting_selection_source": "rule_fallback",
            "counting_rejected_candidates": [],
        }
        selected_candidates = None
        if self.config.detection.use_vlm_refinement and self.vlm_model is not None:
            vlm_candidates, vlm_response, vlm_indices = select_count_candidates_with_vlm(
                self.vlm_model,
                image_pil,
                object_name,
                candidates,
                question=question,
                save_dir=save_dir if visualize else None,
            )
            counting_info["counting_vlm_response"] = vlm_response
            counting_info["counting_vlm_selected_indices"] = vlm_indices or []
            if vlm_candidates:
                filtered_candidates, rejected = filter_count_instance_candidates(
                    vlm_candidates,
                    candidates,
                    object_name,
                    max_instances=max_instances,
                )
                counting_info["counting_rejected_candidates"] = rejected
                if filtered_candidates:
                    selected_candidates = filtered_candidates
                    counting_info["counting_selection_source"] = "vlm_multi_select"

        if selected_candidates is None:
            rule_candidates = select_count_instance_candidates(
                candidates,
                max_instances=max_instances,
                score_margin=0.55,
            )
            selected_candidates, rejected = filter_count_instance_candidates(
                rule_candidates,
                candidates,
                object_name,
                max_instances=max_instances,
            )
            counting_info["counting_rejected_candidates"] = rejected

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
                    "counting_selection_source": counting_info["counting_selection_source"],
                    "counting_vlm_response": counting_info["counting_vlm_response"],
                    "counting_vlm_selected_indices": counting_info["counting_vlm_selected_indices"],
                    "counting_filter_reason": detection.get("counting_filter_reason"),
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
                "counting_selection_source": counting_info["counting_selection_source"],
                "counting_vlm_response": counting_info["counting_vlm_response"],
                "counting_vlm_selected_indices": counting_info["counting_vlm_selected_indices"],
                "counting_filter_reason": detection.get("counting_filter_reason"),
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

        return instances, instance_names, depth, counting_info


    def _select_detection(
        self,
        image_pil: Image.Image,
        object_name: str,
        candidates: List[Dict[str, object]],
        selected_boxes: Dict[str, List[int]],
        save_dir: Optional[Union[str, Path]],
        question: str,
    ) -> Dict[str, object]:
        rule_ranked = rank_candidates_for_object(image_pil, object_name, candidates, selected_boxes)
        rule_selected = dict(rule_ranked[0])
        rule_selected["_ranked_candidates"] = rule_ranked
        rule_selected["rule_selected_index"] = 0
        rule_selected["vlm_selected_index"] = None
        rule_selected["final_selected_index"] = 0
        rule_selected["selection_decision"] = "rule_ranker"
        rule_selected["selection_reject_reason"] = None

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
                rule_selected["selection_reject_reason"] = "invalid_vlm_response"
            else:
                allowed, reject_reason = validate_vlm_selection(object_name, selected, rule_ranked)
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
    prompt_target = target or original
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

    if relation in {"leftmost", "rightmost", "center", "topmost", "bottommost"}:
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
    large_complete_entity = _is_large_complete_entity_target(target)
    exact_large_entity_prompt = large_complete_entity and _prompt_matches_target_for_scoring(prompt, original, target)
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
        if area_ratio > 0.22:
            if exact_large_entity_prompt:
                score -= 0.25 * area_ratio
                reasons.append("large_entity_tolerated")
            else:
                score -= 1.5 * area_ratio
                reasons.append("large_entity_penalty")
        if width_ratio > 0.65 or height_ratio > 0.70:
            if exact_large_entity_prompt:
                score -= 0.08
                reasons.append("broad_entity_tolerated")
            else:
                score -= 0.35
                reasons.append("broad_entity_penalty")
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
    candidate_metadata = format_candidate_metadata_for_vlm(candidates, image_pil.size)
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

Coordinate and score notes:
- box2d is [x1, y1, x2, y2] in image coordinates.
- Larger x means further right in the image; larger y means lower in the image.
- area_ratio is the fraction of the image covered by the candidate box.
- dino_score, rank_score, and rank_reasons are supporting evidence only.
- Do not choose a candidate only because it has the highest score if it violates the target category, attribute, relation, or reference object context.

Candidate 0 is the rule-ranked best candidate, but you may choose another candidate if it better matches the exact target object category and question context.

Selection rules:
- Select the candidate that corresponds to the object referred to in the question, not just the most visually obvious object of the category.
- Use spatial/contextual clues in the question, such as color, material, shape, relative position, nearby objects, and role in the scene.
- If the question distinguishes similar objects, choose the candidate matching the distinguishing phrase, such as "white coffee table", "circular table", "black table", "person wearing a hat", "left chair", or "closer sofa".
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
- For "glass table", choose the actual glass/transparent table, not a black plastic table or ordinary dark table.
- For "translucent cube", choose the transparent/translucent cube, not an opaque box or ordinary cube-like object.
- For "circular table" or "round table", choose the round/circular table, not a rectangular table.
- Do not choose a visually different material, color, transparency, or shape just because it has a higher detector score.

Relation modifier rules:
- Treat words such as rightmost, leftmost, topmost, and bottommost as part of the target object phrase, not as optional context.
- For "rightmost chair" or similar targets, choose the rightmost complete candidate among candidates that match the target category.
- Do not choose a larger, clearer, or more central candidate if it violates the explicit rightmost/leftmost/topmost/bottommost modifier.

Counting and instance rules:
- For counting targets such as handles, curtains, frames, shelves, shoeboxes, lightbulbs, towels, plates, and post-it notes, select candidates that tightly cover one visible instance.
- Do not select the same instance twice.
- Do not select a large box that contains multiple counted instances when a tighter single-instance candidate exists.
- Do not use a container or support object such as a cabinet, door, basket, bed, or shelf as the counted item.
- If a candidate mostly covers background, floor, or several unrelated objects, return INVALID unless the target is an area object.

Badcase-guided examples:
- Good: for "circular table under the TV", choose the circular/round table below the TV. Bad: choosing the TV because it is the reference object.
- Good: for "translucent cube", choose the transparent or semi-transparent cube. Bad: choosing an opaque box, a tiny unrelated part, or a higher-score non-translucent candidate.
- Good: for "gray chair" versus "black chair", choose the candidate matching both category and color. Bad: merging all chairs into one ordinary chair.
- Good: for "brown chair" versus "black chair", color is part of the target phrase and must be respected.
- Good: for "rightmost" or "leftmost" targets, choose the candidate satisfying the explicit image-side modifier. Bad: choosing a clearer but more central object.
- Good: for "closest to the camera" or "furthest from the camera", use full-image context and candidate location as cues; scores are only supporting evidence.
- Good: camera is viewpoint, not a graph/object target. Bad: selecting a physical-looking camera candidate when the question says "from the camera's perspective".
- Good: for "striped balls" versus "solid balls", "polka-dot chair", "glass table", and "circular table", preserve the visual attribute in the target phrase.

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


def select_count_candidates_with_vlm(
    vlm_model: Any,
    image_pil: Image.Image,
    object_name: str,
    candidates: List[Dict[str, object]],
    question: str = "",
    save_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[List[Dict[str, object]]], Optional[str], List[int]]:
    grid = save_candidate_grid(image_pil, object_name, candidates, save_dir)
    overlay_path = save_candidate_overlay(image_pil, object_name, candidates, save_dir)
    overlay_image = Image.open(overlay_path).convert("RGB") if overlay_path is not None else make_candidate_overlay(image_pil, candidates)
    candidate_metadata = format_candidate_metadata_for_vlm(candidates, image_pil.size)
    prompt = f"""
Select every numbered candidate box that tightly covers one visible instance of the count target.

Count target: {object_name}
Question context: {question}

Candidate metadata table:
{candidate_metadata}

Coordinate and score notes:
- box2d is [x1, y1, x2, y2] in image coordinates.
- Larger x means further right in the image; larger y means lower in the image.
- dino_score, rank_score, and rank_reasons are supporting evidence only.
- Prefer all true visible instances over only the clearest or highest-score candidates.

Counting selection rules:
- Return all candidate indices that correspond to separate visible instances of the count target.
- Do not select the same instance twice.
- Do not select a large box that contains multiple counted instances when tighter single-instance candidates exist.
- Do not select a container or support object as the counted item, such as cabinet, door, basket, bed, shelf, chandelier, or fridge, unless that container is itself the count target.
- Do not select mostly background, floor, wall, or a whole group box as a single counted item.
- Use the full-image overlay to check whether candidates are duplicates or group boxes.
- Use the crop grid only as supporting evidence for object identity and box quality.

Badcase-guided counting examples:
- Plates: select all visible plates, not only the clearest one or two.
- Lightbulbs: select individual bulbs; do not select the entire chandelier as a lightbulb.
- Post-it notes: select individual notes; do not select one large box containing several notes.
- Towels: select each towel once; overlapping boxes for the same towel should not both be selected.
- Handles/dials: if the question says dials count as handles, select dials as handle instances too.
- Letters: select individual letters; do not select an entire word or text block as one letter.
- Computer mice: select every visible computer mouse instance; do not keep only one aggregate mouse box.

Return format:
- Return only a Python-style list of integer indices, for example [0, 2, 5].
- If none of the numbered candidates are valid count instances, return INVALID.
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
    response = vlm_model.process_messages(messages, max_new_tokens=128)
    selected_indices = parse_count_candidate_indices(response, len(candidates))
    if not selected_indices:
        return None, str(response), []
    selected = []
    for idx in selected_indices:
        item = dict(candidates[idx])
        item["candidate_index"] = idx
        selected.append(item)
    return selected, str(response), selected_indices


def parse_count_candidate_indices(response: object, num_candidates: int) -> Optional[List[int]]:
    text = str(response or "")
    if "INVALID" in text.upper():
        return None
    match = re.search(r"\[([^\]]*)\]", text)
    if not match:
        return None
    result = []
    for value in re.findall(r"\d+", match.group(1)):
        idx = int(value)
        if 0 <= idx < num_candidates and idx not in result:
            result.append(idx)
    return result or None


def vlm_selection_is_plausible(object_name: str, selected: Dict[str, object], candidates: List[Dict[str, object]]) -> bool:
    return validate_vlm_selection(object_name, selected, candidates)[0]


def validate_vlm_selection(object_name: str, selected: Dict[str, object], candidates: List[Dict[str, object]]) -> Tuple[bool, Optional[str]]:
    if not candidates:
        return False, "no_candidates"
    selected_score = float(selected.get("rank_score", selected.get("score", 0.0)))
    best_score = float(candidates[0].get("rank_score", candidates[0].get("score", 0.0)))
    if _candidate_matches_reference_object(object_name, selected):
        return False, "reference_object_selected"
    if _weak_phrase_low_score_selection(object_name, selected, candidates):
        return False, "weak_phrase_low_score"
    if not _selection_matches_relation(object_name, selected, candidates):
        return False, "relation_mismatch"
    if not is_area_object(object_name):
        selected_area = box_area(selected.get("box2d", [0, 0, 0, 0]))
        best_area = max(1, box_area(candidates[0].get("box2d", [0, 0, 0, 0])))
        if selected_score < best_score and selected_area < best_area * 0.70 and _prefers_complete_entity(object_name):
            if not (
                _relation_target_candidate_override(object_name, selected)
                or _attribute_target_candidate_override(object_name, selected)
            ):
                return False, "partial_entity_box"
        if selected_score < best_score and selected_area > best_area * 2.0:
            if not (_attribute_target_candidate_override(object_name, selected) and selected_area <= best_area * 4.0):
                return False, "entity_box_too_large"
    return True, None


def format_candidate_metadata_for_vlm(candidates: List[Dict[str, object]], image_size: Tuple[int, int]) -> str:
    width, height = image_size
    image_area = float(max(1, width * height))
    lines = [
        "index | prompt | box2d | center | area_ratio | dino_score | rank_score | rank_reasons"
    ]
    for idx, candidate in enumerate(candidates):
        box = clamp_box_to_image(candidate.get("box2d", [0, 0, 0, 0]), image_size)
        cx, cy = box_center(box)
        area_ratio = box_area(box) / image_area
        dino_score = float(candidate.get("score", 0.0))
        rank_score = float(candidate.get("rank_score", dino_score))
        reasons = ",".join(str(reason) for reason in candidate.get("rank_reasons", [])) or "none"
        prompt = str(candidate.get("prompt", "")).replace("\n", " ").strip()
        lines.append(
            f"{idx} | {prompt} | {box} | ({cx:.1f},{cy:.1f}) | "
            f"{area_ratio:.4f} | {dino_score:.3f} | {rank_score:.3f} | {reasons}"
        )
    return "\n".join(lines)


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
    return warnings


def summarize_candidates(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary = []
    for candidate in candidates:
        box2d = [int(v) for v in candidate.get("box2d", [])]
        item = {
            "prompt": candidate.get("prompt", ""),
            "box2d": box2d,
        }
        if len(box2d) == 4:
            cx, cy = box_center(box2d)
            item["center"] = [float(cx), float(cy)]
            item["box_area"] = int(box_area(box2d))
        if "score" in candidate:
            item["score"] = float(candidate["score"])
            item["dino_score"] = float(candidate["score"])
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


def _remove_relation_words(prompt: str) -> str:
    prompt = prompt.replace("left-most", "leftmost").replace("top-most", "topmost")
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


NON_VISUAL_COUNT_RE = re.compile(r"\b(?:need|needed|stack|stacked|achieve|match|reach|same height|have to)\b")


def is_visual_count_target(question_type: Optional[str], question: str, object_name: str) -> bool:
    if question_type != "numeric_ct":
        return False
    question_text = re.sub(r"\s+", " ", str(question or "").lower())
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


def select_count_instance_candidates(
    candidates: List[Dict[str, object]],
    max_instances: int = 5,
    score_margin: float = 0.35,
) -> List[Dict[str, object]]:
    if not candidates:
        return []
    best_score = float(candidates[0].get("rank_score", candidates[0].get("score", 0.0)))
    selected = []
    for candidate in candidates:
        score = float(candidate.get("rank_score", candidate.get("score", 0.0)))
        if score < best_score - score_margin:
            continue
        box = candidate.get("box2d", [0, 0, 0, 0])
        if any(box_iou(box, item.get("box2d", [0, 0, 0, 0])) > 0.75 for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_instances:
            break
    selected.sort(key=lambda item: (box_center(item.get("box2d", [0, 0, 0, 0]))[1], box_center(item.get("box2d", [0, 0, 0, 0]))[0]))
    return selected


def filter_count_instance_candidates(
    selected_candidates: List[Dict[str, object]],
    candidate_pool: List[Dict[str, object]],
    object_name: str,
    max_instances: int = 20,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    kept = []
    rejected = []
    for candidate in selected_candidates:
        reason = count_candidate_reject_reason(candidate, kept, candidate_pool, object_name)
        if reason is not None:
            rejected.append(_summarize_rejected_count_candidate(candidate, reason))
            continue
        item = dict(candidate)
        item["counting_filter_reason"] = None
        kept.append(item)
        if len(kept) >= max_instances:
            break
    kept.sort(key=lambda item: (box_center(item.get("box2d", [0, 0, 0, 0]))[1], box_center(item.get("box2d", [0, 0, 0, 0]))[0]))
    return kept, rejected


def count_candidate_reject_reason(
    candidate: Dict[str, object],
    kept: List[Dict[str, object]],
    candidate_pool: List[Dict[str, object]],
    object_name: str,
) -> Optional[str]:
    box = candidate.get("box2d", [0, 0, 0, 0])
    if any(box_iou(box, item.get("box2d", [0, 0, 0, 0])) > 0.75 for item in kept):
        return "duplicate_iou"
    if _count_candidate_is_container(candidate, object_name):
        return "container_object"
    if _candidate_contains_multiple_smaller_boxes(candidate, candidate_pool):
        return "contained_group_box"
    return None


def _count_candidate_is_container(candidate: Dict[str, object], object_name: str) -> bool:
    prompt = singularize_count_name(_remove_leading_color(_remove_relation_words(str(candidate.get("prompt", "")).lower())))
    target_forms = set(_object_name_forms(object_name))
    target_forms.update(singularize_count_name(form) for form in list(target_forms))
    containers = {"cabinet", "door", "basket", "bed", "shelf", "chandelier", "fridge", "refrigerator"}
    return prompt in containers and prompt not in target_forms


def _candidate_contains_multiple_smaller_boxes(candidate: Dict[str, object], candidate_pool: List[Dict[str, object]]) -> bool:
    box = candidate.get("box2d", [0, 0, 0, 0])
    candidate_area = max(1, box_area(box))
    contained = 0
    for other in candidate_pool:
        other_box = other.get("box2d", [0, 0, 0, 0])
        if _same_box(box, other_box):
            continue
        other_area = box_area(other_box)
        if other_area <= 0 or other_area >= candidate_area * 0.65:
            continue
        if box_center_inside(other_box, box):
            contained += 1
        if contained >= 2:
            return True
    return False


def _summarize_rejected_count_candidate(candidate: Dict[str, object], reason: str) -> Dict[str, object]:
    item = summarize_candidates([candidate])[0]
    item["counting_filter_reason"] = reason
    return item


def relation_modifier(object_name: str) -> str:
    name = object_name.lower().replace("left-most", "leftmost").replace("top-most", "topmost")
    for modifier in ["leftmost", "rightmost", "center", "topmost", "bottommost"]:
        if name.startswith(modifier + " "):
            return modifier
    return ""


def select_by_relation(ranked, relation: str, width: int, height: int) -> Dict[str, object]:
    candidates = _reasonable_relation_candidates([item[2] for item in ranked], relation)
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


def _selection_matches_relation(object_name: str, selected: Dict[str, object], candidates: List[Dict[str, object]]) -> bool:
    relation = relation_modifier(object_name)
    if relation not in {"leftmost", "rightmost", "topmost", "bottommost"}:
        return True
    reasonable = _reasonable_relation_candidates(candidates, relation)
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
    return True


def is_area_object(object_name: str) -> bool:
    base = _remove_leading_color(_remove_relation_words(object_name.lower()))
    return base in {"carpet", "rug", "floor", "wall", "ceiling"}


def _prefers_complete_entity(object_name: str) -> bool:
    base = _remove_leading_color(_remove_relation_words(object_name.lower()))
    return any(word in base for word in ("cabinet", "table", "sofa", "chair"))


def _is_large_complete_entity_target(object_name: str) -> bool:
    base = _remove_leading_color(_remove_relation_words(object_name.lower()))
    return any(word in base for word in ("bed", "sofa", "couch", "fireplace", "cabinet", "bookshelf", "bookcase", "wardrobe"))


def _prompt_matches_target_for_scoring(prompt: str, original: str, target: str) -> bool:
    prompt = prompt.lower().strip()
    if not prompt:
        return False
    if prompt == original or prompt == target:
        return True
    return prompt in set(detection_prompts_for_object(target))


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


def _weak_phrase_low_score_selection(object_name: str, selected: Dict[str, object], candidates: List[Dict[str, object]]) -> bool:
    if not candidates:
        return False
    best = candidates[0]
    if _candidate_matches_target_phrase(object_name, selected):
        return False
    if not _candidate_matches_target_phrase(object_name, best):
        return False
    target = parse_object_relation_context(object_name).get("target_phrase", "")
    selected_prompt = str(selected.get("prompt", "")).lower().strip()
    if not target or not selected_prompt:
        return False
    target_words = set(re.findall(r"[a-z0-9]+", target.lower()))
    selected_words = set(re.findall(r"[a-z0-9]+", selected_prompt))
    if not selected_words or not selected_words.issubset(target_words):
        return False
    if len(selected_words) >= len(target_words):
        return False
    selected_rank = float(selected.get("rank_score", selected.get("score", 0.0)))
    best_rank = float(best.get("rank_score", best.get("score", 0.0)))
    selected_dino = float(selected.get("score", 0.0))
    best_dino = float(best.get("score", 0.0))
    return best_dino >= 0.45 and selected_dino <= max(0.15, best_dino * 0.35) and selected_rank <= best_rank - 0.35


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
