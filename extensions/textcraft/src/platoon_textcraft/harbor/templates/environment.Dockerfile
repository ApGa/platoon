FROM @@BASE_IMAGE@@
# The image is the same for every task (and for the verifier): Python, tmux for terminal agents, and the
# `textcraft` command, which runs the game file below.
RUN @@IMAGE_RUN@@
# The game and the world's recipes (the same in every task), then this task's goal and starting inventory.  The
# game keeps its action log next to them.
WORKDIR /app
ENV TEXTCRAFT_DIR=/app/textcraft
COPY textcraft_game.py recipes.json /app/textcraft/
COPY task.json /app/textcraft/
