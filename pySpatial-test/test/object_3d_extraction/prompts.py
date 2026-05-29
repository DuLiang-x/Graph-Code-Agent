"""Prompt templates for object 3D extraction."""

PROMPT_GET_OBJECTS_OF_INTEREST = """
### Situation Description
Given an image and a spatial reasoning question, we need to all entities that are included in the question.

Camera rule:
- In Omni3D-Bench single-image tasks, "camera" usually means the viewpoint of the current image, not a visible object.
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
- Do not replace "stool" with "chair".
- Do not replace "chair" with "stool".
- Do not replace "bench", "sofa", "couch", "ottoman", or "seat" with "chair" unless the question itself uses that word.
- Do not include relation words such as left, right, closer, farther, above, below, front, behind.
- Do not include "camera" unless it is a visible physical camera object.

[Question] {question}
[Detect]
"""

PATTERN_GET_OBJECTS_OF_INTEREST = r"\[[^\]]*\]"
