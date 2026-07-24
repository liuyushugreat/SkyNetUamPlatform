# Clean-room reproduction environment for the SkyRwa artifact.
#
# Build (from modules/SkyRwa/):
#   docker build -t skyrwa-artifact .
#
# Run the full reproduction pipeline (~10 min):
#   docker run --rm skyrwa-artifact
#
# Or open a shell to run individual scripts:
#   docker run --rm -it skyrwa-artifact bash

FROM python:3.12-slim

# The package is imported as `SkyRwa`, so it must live one level below
# the working directory.
WORKDIR /app/modules
COPY . /app/modules/SkyRwa

RUN pip install --no-cache-dir -r /app/modules/SkyRwa/requirements.txt

# Sanity check: test suite must pass in the clean room.
RUN python -m pytest SkyRwa/tests -q

WORKDIR /app/modules/SkyRwa/ISWC2026
CMD ["bash", "run.sh"]
