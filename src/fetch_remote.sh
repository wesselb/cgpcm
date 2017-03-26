#!/usr/bin/env bash
rsync -aPz aws:/home/ec2-user/icml/cgpcm/src/tasks/cache tasks/remote
rsync -aPz aws:/home/ec2-user/icml/cgpcm/src/output output/remote
