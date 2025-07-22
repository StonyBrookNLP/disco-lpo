#!/bin/bash
CODEQL="$HOME/path_to_current_folder/codeql/bin/codeql"
cd $1
$CODEQL database create "./database" --overwrite --language=python
$CODEQL database analyze "./database" /path_to_current_folder/codeql/codeql-repo/python/ql/src/Security --format=csv --output="./codeql_analysis.csv";