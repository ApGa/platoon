FROM @@BASE_IMAGE@@
# The same image as the task's environment, so one image serves the agent and the verifier.
RUN @@IMAGE_RUN@@
# The verifier's trusted copies of the grader, the game, the recipes and the task.
WORKDIR /app
COPY test.sh textcraft_game.py recipes.json task.json /tests/
