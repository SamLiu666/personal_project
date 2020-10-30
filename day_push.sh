#!/bin/bash
git status
git add -A
echo -r " Please Enter Your Push Record ->"
read record
echo -n
git commit -m $record
git push test master

echo "done"
