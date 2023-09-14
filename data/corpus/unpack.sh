#!/bin/bash

source_directory="./"

if [ ! -d "$source_directory" ]; then
  echo "The source directory '$source_directory' does not exist."
  exit 1
fi

for file in "$source_directory"/*.xml.gz; do
  if [ -f "$file" ]; then
    gunzip "$file"
    echo "Decompressed file: $file"
  else
    echo "File '$file' does not exist."
  fi
done

echo "Decompression completed."
