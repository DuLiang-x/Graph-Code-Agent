"""Prompt templates for object 3D extraction."""

QUESTION_TYPE_RULES = {
    "count_ratio": """
Question type: counting ratio.
Counting-ratio rules:
- The question asks for a ratio between counts of visible object categories, not a geometric size ratio.
- Detect every countable category in the ratio and preserve attributes such as black, white, TV, or remote.
- The extraction pipeline will expand count targets into indexed instances such as coaster_1 and black_tv_remote_1 after detection.
- Do not collapse plural count targets into one aggregate object.

# Example: ratio of two count categories
[Question] What is the ratio of coasters to black TV remotes? Answer with a decimal
[Detect] [coasters, black TV remotes]
""",
    "numeric_ct": """
Question type: numeric count.
Count-question rules:
- Detect all visible instances of the countable object category requested by the question, not only one example instance.
- If the question asks how many handles, lights, curtains, shoeboxes, shelves, windows, chairs, stools, or other instances are visible, include that countable object category in [Detect].
- Use natural object category names such as handles, lights, or chairs in [Detect]; do not invent indexed names such as handle_1 or chair_2 in the detection list.
- Include supporting or reference objects only when they are needed to locate or disambiguate the counted instances.
- For count questions, the extraction pipeline will assign instance IDs such as handle_1, handle_2 after detection; your [Detect] output should still be [handles, cabinets].
- Do not convert plural count targets into a different broad object category.
- For generic questions like "how many objects/items/things are stuck, attached, or on a reference surface", output one unified count target plus the reference object, for example [objects stuck on fridge, fridge]. Do not split the same visible items into overlapping categories such as magnets, pictures, notes, and stickers.
- If the question says "dials count as handles", include both dials and handles so both can be counted.

# Example: count handles on cabinets
[Question] How many handles are on the cabinets?
[Detect] [handles, cabinets]

# Example: count visible shoeboxes
[Question] How many shoeboxes are visible on the shelf?
[Detect] [shoeboxes, shelf]

# Example: count all objects stuck on a reference object
[Question] Including all objects, how many objects are stuck on the fridge?
[Detect] [objects stuck on fridge, fridge]

# Example: dials count as handles
[Question] Counting the dials as handles, how many handles are shown?
[Detect] [dials, handles]
""",
    "numeric_other": """
Question type: numeric measurement or ratio.
Numeric-measurement rules:
- Detect every object used as a numeric operand in the calculation.
- For ratio, difference, sum, combined height, width, length, depth, distance, or volume questions, include all numerator, denominator, and reference objects.
- For known-size calibration questions, include both the object with the provided size and the object whose size is requested.
- Questions like "How many objects with the volume/height/width/length of X would fit, stack, reach, or match Y" are numeric ratio questions, not visual counting questions.
- For fit/stack/reach/match numeric questions, detect the measurement operands such as X and Y; do not treat them as a request to count visible instances.
- For combined operands such as "two bedside tables" or "combined volume", include the combined object phrase and the comparison object, for example [bedside tables, bed].
- Preserve relation modifiers such as rightmost, leftmost, topmost, bottommost, under, above, or next to when they identify which instance is needed.
- Preserve color and material attributes such as white, black, glass, wooden, or metal when they disambiguate the target object.

# Example: height ratio with combined denominator
[Question] What is the ratio of the height of the fireplace to the combined height of the coffee table and the sofa to the right of the coffee table?
[Detect] [fireplace, coffee table, sofa]

# Example: known-size calibration
[Question] If the black table is 1.5m wide, how tall is the TV?
[Detect] [black table, TV]

# Example: volume ratio with combined operand
[Question] How many objects with the volume of the combined volume of the two bedside tables would fit in an object with the volume of the bed?
[Detect] [bedside tables, bed]

# Example: height ratio phrased as how many objects
[Question] How many objects of the same height as the TV would reach the height of the sofa?
[Detect] [TV, sofa]
""",
    "yes_no": """
Question type: yes/no relation, visibility, or collision.
Yes/no-question rules:
- Detect all objects needed to evaluate the relation, visibility, support, containment, collision, or hypothetical movement.
- If the question asks whether X would block, hit, cover, fit on, contain, support, or be visible relative to Y, include both X and Y.
- Include intermediate reference objects when the relation depends on them.
- Do not include camera when it means the image viewpoint.

# Example: visibility after hypothetical placement
[Question] If the sofa were placed in front of the fireplace, would the fireplace still be visible?
[Detect] [sofa, fireplace]

# Example: collision target
[Question] If the TV were to fall forward, would it hit the lamp or the glass table first?
[Detect] [TV, lamp, glass table]
""",
    "choice_object": """
Question type: object choice or direction choice.
Choice-question rules:
- Detect every candidate object and every reference object needed to compare the candidates.
- If the question provides Options:{...} or says choose from a list, every physical object in the options must be included in [Detect].
- For questions asking which object is closer, farther, left, right, front, behind, above, below, or first hit, include all compared objects and the reference object.
- Direction labels such as N, NE, E, SE, S, SW, W, NW are answers, not detectable objects.
- Do not invent objects outside the candidates and references in the question.

# Example: object-choice with options
[Question] Which object is closer to the fireplace: the sofa or the white coffee table? Options: {sofa, coffee table}
[Detect] [fireplace, sofa, white coffee table, coffee table]

# Example: object-choice without explicit options
[Question] What is closer to the camera: the small tree or the leftmost lamp?
[Detect] [small tree, leftmost lamp]
""",
    "generic": """
Question type: generic spatial question.
Generic rules:
- Detect all physical objects explicitly needed to answer the question.
- Keep object phrases faithful to the wording of the question.
""",
}

PROMPT_GET_OBJECTS_OF_INTEREST = """
### Situation Description
Given an image and a spatial reasoning question, we need to all entities that are included in the question.

Camera rule:
- In Omni3D-Bench single-image tasks, "camera" usually means the viewpoint of the current image, not a visible object.
- Do not include "camera" in [Detect] when it refers to the image/camera perspective, including "facing the camera", "towards the camera", "closer to the camera", and "from the camera".
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

{question_type_rules}

### Your Task
Now, given the question below, please identify the entities that are included in the question.

[Question] {question}
[Detect]
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
- For counting questions, keep the countable category as a natural plural/category phrase such as "handles" or "chairs"; do not invent indexed names such as "handle_1" in [Detect].
- For generic surface counting such as "objects/items/things stuck on the fridge", use one unified count target such as "objects stuck on fridge" plus the reference object; do not split the same items into multiple overlapping categories.
- Example repair target: objects stuck on the fridge -> [objects stuck on fridge, fridge].
- Do not collapse a count target into a support object such as cabinet, shelf, table, or wall.
- Do not replace "stool" with "chair".
- Do not replace "chair" with "stool".
- Do not replace "bench", "sofa", "couch", "ottoman", or "seat" with "chair" unless the question itself uses that word.
- Do not include relation words such as left, right, closer, farther, above, below, front, behind.
- Do not include "camera" unless it is a visible physical camera object.
- For "dials count as handles", keep both "dials" and "handles".

[Question] {question}
[Detect]
"""

PATTERN_GET_OBJECTS_OF_INTEREST = r"\[[^\]]*\]"
