"""Prompt templates for object 3D extraction."""

QUESTION_TYPE_RULES = {
    "number_ct": """
Question type: number visual counting.
Count-question rules:
- Use these rules only when the question asks for the number/count of visible instances in the image.
- Plural words alone do not mean the task is visual counting.
- Do not treat "How many of X would you stack/reach/match/fit..." as visual counting; that is a numeric ratio/measurement question when the answer type is float.
- Detect all visible instances of the countable object category requested by the question, not only one example instance.
- If the question asks how many handles, lights, curtains, shoeboxes, shelves, windows, chairs, stools, or other instances are visible, include that countable object category in [Detect].
- Use natural object category names such as handles, lights, or chairs in [Detect]; do not invent indexed names such as handle_1 or chair_2 in the detection list.
- Include supporting or reference objects only when they are needed to locate or disambiguate the counted instances.
- For count questions, the extraction pipeline will assign instance IDs such as handle_1, handle_2 after detection; your [Detect] output should still be [handles, cabinets].
- Do not convert plural count targets into a different broad object category.

# Example: count handles on cabinets
[Question] How many handles are on the cabinets?
[Detect] [handles, cabinets]

# Example: count visible shoeboxes
[Question] How many shoeboxes are visible on the shelf?
[Detect] [shoeboxes, shelf]

# Example: count-ratio with attributed categories
[Question] What is the ratio of brown chairs to black chairs? Answer as a decimal.
[Detect] [brown chairs, black chairs]
""",
    "number_other": """
Question type: number measurement or non-counting ratio.
Numeric-measurement rules:
- Detect every object used as a numeric operand in the calculation.
- Plural words alone do not mean the task is visual counting.
- For "How many of X would you stack/reach/match the height of Y", detect the exact X and Y operands and do not convert them into plural count categories.
- For "How many objects with the volume/height/width/length of X would fit/reach Y", treat it as a continuous numeric ratio/measurement question, not visual counting.
- For ratio, difference, sum, combined height, width, length, depth, distance, or volume questions, include all numerator, denominator, and reference objects.
- For known-size calibration questions, include both the object with the provided size and the object whose size is requested.
- Compound object names such as TV stand are independent physical objects; keep TV and TV stand separate when both are numeric operands.
- Preserve relation modifiers such as rightmost, leftmost, topmost, bottommost, center, middle, closest, furthest, upper, lower, under, above, or next to when they identify which instance is needed.
- Same-category objects with different instance modifiers are different numeric operands and must both be detected, such as leftmost cabinet and center cabinet.
- Same base-category operands with different relation contexts are different operands and must both be detected, such as cabinets to the left of a fume vent and cabinet to the right of the fume vent.
- Preserve color and material attributes such as white, black, glass, wooden, or metal when they disambiguate the target object.
- If the question says two X, both X, multiple X, all X, or combined height/width/length/depth/volume of two X, include the natural plural/category phrase so the pipeline can detect each instance separately.
- Count-ratio questions such as "ratio of coasters to remotes" or "ratio of brown chairs to black chairs" require all visible instances on both sides of the ratio; include both countable categories with their attributes.

# Example: height ratio with combined denominator
[Question] What is the ratio of the height of the fireplace to the combined height of the coffee table and the sofa to the right of the coffee table?
[Detect] [fireplace, coffee table, sofa]

# Example: known-size calibration
[Question] If the black table is 1.5m wide, how tall is the TV?
[Detect] [black table, TV]

# Example: compound object operand is distinct from its head noun
[Question] What is the ratio of the height of the TV to the width of the TV stand?
[Detect] [TV, TV stand]

# Example: same-category operands with different relation contexts
[Question] If the width of the combined cabinets to the left of the fume vent is 4.2m, how tall is the cabinet to the right of the fume vent in meters?
[Detect] [cabinets to the left of the fume vent, cabinet to the right of the fume vent, fume vent]

# Example: same-category operands with different instance modifiers
[Question] What is the ratio of the height of the leftmost cabinet to the width of the center cabinet?
[Detect] [leftmost cabinet, center cabinet]

# Example: known-height calibration with attributed source object
[Question] If the 3D height of the wooden chair is 3.80 meters, what is the 3D height of the table in meters?
[Detect] [wooden chair, table]

# Example: known-height calibration with different target object
[Question] If the 3D height of the fridge is 4.80 meters, what is the 3D height of the chair in meters?
[Detect] [fridge, chair]

# Example: same-height stack is numeric ratio, not visual counting
[Question] How many objects of the same height as the armchair would I need to make a structure as tall as the dresser closest to the camera?
[Detect] [armchair, dresser]

# Example: two same-category numeric operands
[Question] What is the combined width of the two sinks compared with the bathtub?
[Detect] [sinks, bathtub]

# Example: count-ratio numeric operands
[Question] What is the ratio of coasters to black TV remotes?
[Detect] [coasters, black TV remotes]

# Example: combined volume of repeated objects
[Question] How many objects with the combined volume of two bedside tables fit in the bed?
[Detect] [bedside tables, bed]

# Example: preposition reference object is a separate numeric operand
[Question] Is the chair closer to the stool than the table?
[Detect] [chair, stool, table]

# Example: attributed objects around a reference relation
[Question] Is the white chair next to the black stool?
[Detect] [white chair, black stool]

# Example: relation reference operands should be separate objects
[Question] Is the chair closer to the stool than the table?
[Detect] [chair, stool, table]

# Example: stack/reach height is not visual counting
[Question] How many of the rightmost stool would you have to stack to reach the same height as the left-most chair?
[Detect] [rightmost stool, leftmost chair]
""",
    "yes_no": """
Question type: yes/no relation, visibility, or collision.
Yes/no-question rules:
- Detect all objects needed to evaluate the relation, visibility, support, containment, collision, or hypothetical movement.
- If the question asks whether X would block, hit, cover, fit on, contain, support, or be visible relative to Y, include both X and Y.
- If the yes/no question compares counts, such as whether the number of X is greater than the number of Y, include both countable categories so the pipeline can expand all visible instances.
- Include intermediate reference objects when the relation depends on them.
- Do not include camera when it means the image viewpoint.
- If the question asks whether there are two or more of the same object type, output the concrete repeated category visible in the image, such as [chairs], not [same object types].

# Example: visibility after hypothetical placement
[Question] If the sofa were placed in front of the fireplace, would the fireplace still be visible?
[Detect] [sofa, fireplace]

# Example: relation phrase reference object
[Question] Is the stool to the left of the piano and in front of the piano?
[Detect] [stool, piano]

# Example: enough/each implies counting the distributed objects
[Question] Are there enough fruits in the basket for each person at the table to get one?
[Detect] [fruits, basket, people, table]

# Example: collision target
[Question] If the TV were to fall forward, would it hit the lamp or the glass table first?
[Detect] [TV, lamp, glass table]
""",
    "multi_choice": """
Question type: object choice or direction choice.
Choice-question rules:
- Detect every candidate object and every reference object needed to compare the candidates.
- If the question provides Options:{{...}} or says choose from a list, every physical object in the options must be included in [Detect].
- For questions asking which object is closer, farther, left, right, front, behind, above, below, or first hit, include all compared objects and the reference object.
- If the choice question compares groups by count, such as which option has more/fewer visible X, include each countable option category so the pipeline can expand all visible instances.
- Direction labels such as N, NE, E, SE, S, SW, W, NW are answers, not detectable objects.
- Do not invent objects outside the candidates and references in the question.

# Example: object-choice with options
[Question] Which object is closer to the fireplace: the sofa or the white coffee table? Options: {{sofa, coffee table}}
[Detect] [fireplace, sofa, white coffee table, coffee table]

# Example: object-choice without explicit options
[Question] What is closer to the camera: the small tree or the leftmost lamp?
[Detect] [small tree, leftmost lamp]

# Example: same-category choice operands with different instance modifiers
[Question] Which is closer to the camera, the left chair or the right chair?
[Detect] [left chair, right chair]

# Example: options may be labels for attributed object categories
[Question] Are there more wooden chairs or leather chairs? Options: {{wooden, leather}}
[Detect] [wooden chairs, leather chairs]
""",
    "generic": """
Question type: generic spatial question.
Generic rules:
- Detect all physical objects explicitly needed to answer the question.
- Keep object phrases faithful to the wording of the question.
""",
}

QUESTION_TYPE_RULES["number_vt"] = QUESTION_TYPE_RULES["number_ct"]
QUESTION_TYPE_RULES["numeric_ct"] = QUESTION_TYPE_RULES["number_ct"]
QUESTION_TYPE_RULES["numeric_other"] = QUESTION_TYPE_RULES["number_other"]
QUESTION_TYPE_RULES["choice_object"] = QUESTION_TYPE_RULES["multi_choice"]

PROMPT_GET_OBJECTS_OF_INTEREST = """
### Situation Description
Given an image and a spatial reasoning question, identify only the physical objects whose 3D boxes are required to answer the question.

Camera rule:
- In Omni3D-Bench single-image tasks, "camera" usually means the viewpoint of the current image, not a visible object.
- Camera is the coordinate origin [0, 0, 0] / image viewpoint, not a graph node and not a detectable object.
- Do not include "camera" in [Detect] when it refers to the image/camera perspective.
- Only include "camera" in [Detect] if the question explicitly refers to a visible physical camera object in the scene.

Similar object rule:
- Preserve the exact object category mentioned in the question when objects are visually similar but semantically different.
- Do not replace "stool" with "chair".
- Do not replace "chair" with "stool".
- Do not replace "bench", "sofa", "couch", "ottoman", or "seat" with "chair" unless the question itself uses the word "chair".
- Do not replace "sofa" or "couch" with "chair".
- Do not replace "ottoman" with "stool" or "chair".
- If the question compares two similar objects, include both as separate entities.
- If the question says a generic "seat", keep "seat" as the detection target instead of guessing chair/stool/bench.
- Keep attributes attached to the object phrase, such as "wooden stool", "black chair", "small bench", or "white sofa".

Attribute distinction rule:
- If multiple objects share the same category but differ by color, material, transparency, shape, size, or texture, keep them as separate full object phrases.
- Do not merge "gray chair" and "black chair" into "chair".
- Do not drop attributes such as gray, black, white, glass, wooden, translucent, transparent, clear, circular, round, or square when they identify the target instance.
- If the question compares two attributed objects of the same category, include both attributed phrases separately.

Relation reference rule:
- If a prepositional phrase names a physical reference object needed for relation, comparison, or calculation, include that reference object as a separate [Detect] item.
- Keep the original target phrase in [Detect] when relation context is needed, and decompose it in [Objects]. For example, [Detect] can include "chair at the end of the counter", while [Objects] sets object="chair" and reference_object="counter".
- Keep same-category operands with different instance modifiers separate, such as leftmost cabinet and center cabinet.

Structured object decomposition rule:
- Always output the old [Detect] bracketed list first. [Detect] stores the raw target phrases used as result keys.
- Then output [Objects] as a JSON list. Each item must have: detect_phrase, object, relation_context, reference_object, multi_instance, multi_instance_reason.
- object is the main physical category used for GroundingDINO captions. Do not include relation context in object.
- relation_context and reference_object are only for candidate selection/disambiguation.
- If reference_object is a physical object, include it as its own [Detect] item or let [Objects] reference_object add it automatically.
- For simple objects with no relation, use empty strings for relation_context and reference_object.
- Set multi_instance=true only when this object should be expanded into indexed instances for visual counting, count-ratio, count comparison, or explicit two/both/all/multiple same-category operands.
- Set multi_instance=false for single relation-specific targets such as bottom-most/topmost/rightmost/leftmost X, collision choices such as X first or Y, and continuous numeric stack/reach/fit/height/volume questions.
- Do not set multi_instance=true just because a word contains most, as in bottom-most/topmost/rightmost/leftmost.

Badcase-guided object mention rules:
- Do not output answer-format or math words as objects, such as decimal, sum, direction, square, format, greater, closer, furthest point, or can you fit.
- Do not output color words alone, such as red, white, blue, or black, unless the question explicitly refers to visible color swatches or colored objects as physical candidates.
- Do not output abstract phrases such as same object types, object type, physical objects, answer, or objects required.
- Keep TV and TV stand as different objects; do not replace TV stand with TV or merge them into one target.
- If the question says combined X and Y, or X and Y combined, output the individual physical objects [X, Y], not a synthetic combined object.
- If the question explicitly says two X, both X, multiple X, or combined size/volume of two X, keep the countable category in [Detect] so the pipeline can produce X_1, X_2, etc.
- Do not convert a singular attributed or relation-specific target such as rightmost stool, leftmost chair, black chair, or circular table under the TV into a plural category unless the question truly asks to count visible instances.
- For generic surface counting such as objects stuck on a fridge, keep the counted small-object target and the reference surface; do not collapse the answer to only the fridge.
- For same-type existence questions such as "Are there two of the same object types?", output the concrete repeated physical category visible in the image, such as [chairs] or [lamps], not [same object types].

# Example: camera perspective should not be detected as an object
[Question] From the camera's perspective, is the chair on the left or right of the table?
[Detect] [chair, table]

# Example 1
[Question] You are standing at the airplane's position, facing where it is facing. Is the the person on your left or right?
[Detect] [airplane, person]

# Examples 2
[Question] From the old man's perspective, is the person wearing a hat on the left of the green car?
[Detect] [old man, person wearing a hat, green car]

# Examples 3
[Question] From the car's perspective, which is on the right side: the person or the tree?
[Detect] [car, person, tree]

# Example: chair and stool should not be merged
[Question] Is the chair to the left or right of the stool?
[Detect] [chair, stool]

# Example: keep stool as stool
[Question] Is the stool closer to the table than the chair?
[Detect] [stool, table, chair]

# Example: keep bench distinct from chair
[Question] Which is farther from the camera, the bench or the chair?
[Detect] [bench, chair]

# Example: do not simplify couch/sofa to chair
[Question] Is the sofa behind the chair?
[Detect] [sofa, chair]

# Example: keep ottoman distinct
[Question] Is the ottoman in front of the couch?
[Detect] [ottoman, couch]

# Example: preserve attributes
[Question] Is the black chair closer to the camera than the wooden stool?
[Detect] [black chair, wooden stool]

# Example: same category with different colors should stay separate
[Question] Is the gray chair closer to the table than the black chair?
[Detect] [gray chair, table, black chair]

# Example: preserve transparency attribute
[Question] How many objects of height equal to the height of the translucent cube are needed to match the chair?
[Detect] [translucent cube, chair]

# Example: combined expression should be split into physical objects
[Question] Which is taller in 3D: the sofa or the tv and the tv stand combined? Options: {{sofa, combined tv and tv stand}}
[Detect] [sofa, tv, tv stand]

# Example: same object type asks for repeated concrete categories
[Question] Are there two of the same object types?
[Detect] [chairs]

# Example: structured relation target
[Question] Is the chair at the end of the counter taller than the fireplace?
[Detect] [chair at the end of the counter, fireplace]
[Objects]
[
  {{"detect_phrase":"chair at the end of the counter","object":"chair","relation_context":"at the end of the counter","reference_object":"counter","multi_instance":false,"multi_instance_reason":"single relation-specific chair"}},
  {{"detect_phrase":"fireplace","object":"fireplace","relation_context":"","reference_object":"","multi_instance":false,"multi_instance_reason":"single comparison target"}}
]

# Example: table under TV uses table as GroundingDINO target
[Question] Is the table under the TV wider than the sofa?
[Detect] [table under the TV, sofa]
[Objects]
[
  {{"detect_phrase":"table under the TV","object":"table","relation_context":"under the TV","reference_object":"TV","multi_instance":false,"multi_instance_reason":"single relation-specific table"}},
  {{"detect_phrase":"sofa","object":"sofa","relation_context":"","reference_object":"","multi_instance":false,"multi_instance_reason":"single comparison target"}}
]

# Example: relation-specific cabinet operands
[Question] If the width of the cabinets to the left of the fume vent is 4.2m, how tall is the cabinet to the right of the fume vent?
[Detect] [cabinets to the left of the fume vent, cabinet to the right of the fume vent, fume vent]
[Objects]
[
  {{"detect_phrase":"cabinets to the left of the fume vent","object":"cabinets","relation_context":"to the left of the fume vent","reference_object":"fume vent","multi_instance":true,"multi_instance_reason":"combined cabinets operand may require multiple cabinet instances"}},
  {{"detect_phrase":"cabinet to the right of the fume vent","object":"cabinet","relation_context":"to the right of the fume vent","reference_object":"fume vent","multi_instance":false,"multi_instance_reason":"single relation-specific cabinet"}},
  {{"detect_phrase":"fume vent","object":"fume vent","relation_context":"","reference_object":"","multi_instance":false,"multi_instance_reason":"reference object"}}
]

# Example: central/rightmost same-category targets
[Question] What is the ratio of the height of the central couch to the height of the rightmost couch?
[Detect] [central couch, rightmost couch]
[Objects]
[
  {{"detect_phrase":"central couch","object":"couch","relation_context":"central","reference_object":"","multi_instance":false,"multi_instance_reason":"single relation-specific couch"}},
  {{"detect_phrase":"rightmost couch","object":"couch","relation_context":"rightmost","reference_object":"","multi_instance":false,"multi_instance_reason":"single relation-specific couch"}}
]

# Example: bottom-most is a spatial modifier, not a count trigger
[Question] If the bottom-most frame on the left of the image were to detach from the wall and fall, would it hit the lamp first or the floor?
[Detect] [bottom-most frame on the left of the image, wall, lamp, floor]
[Objects]
[
  {{"detect_phrase":"bottom-most frame on the left of the image","object":"frame","relation_context":"bottom-most on the left of the image","reference_object":"image","multi_instance":false,"multi_instance_reason":"single relation-specific frame"}},
  {{"detect_phrase":"wall","object":"wall","relation_context":"","reference_object":"","multi_instance":false,"multi_instance_reason":"collision reference"}},
  {{"detect_phrase":"lamp","object":"lamp","relation_context":"","reference_object":"","multi_instance":false,"multi_instance_reason":"choice target"}},
  {{"detect_phrase":"floor","object":"floor","relation_context":"","reference_object":"","multi_instance":false,"multi_instance_reason":"choice target"}}
]

# Example: count-ratio targets need multi-instance expansion
[Question] What is the ratio of brown chairs to black chairs? Answer as a decimal.
[Detect] [brown chairs, black chairs]
[Objects]
[
  {{"detect_phrase":"brown chairs","object":"chairs","relation_context":"","reference_object":"","multi_instance":true,"multi_instance_reason":"count-ratio target; count all visible brown chairs"}},
  {{"detect_phrase":"black chairs","object":"chairs","relation_context":"","reference_object":"","multi_instance":true,"multi_instance_reason":"count-ratio target; count all visible black chairs"}}
]

{question_type_rules}

### Your Task
Now, given the question below, please identify the entities that are included in the question.

[Question] {question}
[Detect]
[Objects]
"""

PROMPT_GET_OBJECTS_OF_INTEREST_AUX = """
Looks like your response is not in the correct format!

Previous response: {response}

Please modify your response to the correct format.

Rules:
- Return only one bracketed Python-style list.
- Keep the exact object category used in the question.
- Keep full object phrases such as "black chair", "wooden stool", "white coffee table", or "person wearing a hat".
- Keep color, material, transparency, shape, size, and texture attributes when they identify the target object.
- Do not merge same-category attributed objects such as "gray chair" and "black chair" into "chair".
- Keep transparency phrases such as "translucent cube", "transparent box", or "clear container" intact.
- For true visual counting questions, keep the countable category as a natural plural/category phrase such as "handles" or "chairs"; do not invent indexed names such as "handle_1" in [Detect].
- Do not change singular numeric operands such as "rightmost stool" or "leftmost chair" into plural categories.
- Do not collapse a count target into a support object such as cabinet, shelf, table, or wall.
- Do not replace "stool" with "chair".
- Do not replace "chair" with "stool".
- Do not replace "bench", "sofa", "couch", "ottoman", or "seat" with "chair" unless the question itself uses that word.
- Do not include relation words such as left, right, closer, farther, above, below, front, behind.
- Do not include answer-format or math words such as decimal, sum, direction, square, format, closer, furthest point, or can you fit.
- Do not include abstract phrases such as same object types, object type, or physical objects.
- If your response contains a synthetic combined object such as "combined tv and tv stand", split it into the individual physical objects "tv" and "tv stand".
- Do not include color words alone unless they are visible physical color swatches or colored candidate objects.
- Do not include "camera" unless it is a visible physical camera object.

[Question] {question}
[Detect]
"""

PATTERN_GET_OBJECTS_OF_INTEREST = r"\[[^\]]*\]"
