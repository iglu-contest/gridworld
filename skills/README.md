# Labeling files for embodiment skills

There are 5 skills in total. Each skill describes a kind of gameplay style (action patterns) that the agent need to perform in the environment in order to build each srtucture. We emphasize that skills are properties of the block structures not the agents, in this sense. Here is the list of skills:

  * `flat` - flat structure with all blocks on the ground
  * `flying` - there are blocks that cannot be placed without removing some other blocks (i.e. )
  * `diagonal` - some blocks are adjacent (in vertical axis) diagonally
  * `tricky` - some blocks are hidden or there should be a specific order in which they should be placed
  * `tall` - a structure cannot be built without the agent being high enough (the placement radius is 3 blocks)

The labeling is provided in `skills/skills.yaml` and rendered goal structures are present in `skills/renderings/` folder.
