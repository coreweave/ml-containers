# ml_images

Repository for building ML images at CoreWeave

## Organization
This repository contains multiple container image Dockerfiles, each is expected
to be within its own folder along with any other needed files for the build.

## CI Builds (Actions)
The current CI builds are setup to run when changes to files in the respective
folders are detected so that only the changed container images are built.  The
actions are setup with an action per image utilizing a reusable base action
[build.yml](.github/workflows/build.yml).  The reusable action accepts several inputs:

- `folder` - the folder containing the dockerfile for the image
- `image-name` - the name to use for the image
- `build-args` - arguments to pass to the docker build

Images built using the same source can utilize one action as the main reason for
the multiple actions is to handle only building the changed images.  A build
matrix can be helpful for these cases
https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs.
