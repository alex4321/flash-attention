#!/bin/env bash
compile_directory() {
    DIRECTORY="$1"
    pushd "$DIRECTORY" > /dev/null
    SUM=`find . -type f ! -name "checksum.txt" -exec sha256sum {} \; | sha256sum`
    PREVIOUS_SUM=`cat checksum.txt 2>/dev/null`
    if [[ "$PREVIOUS_SUM" != "$SUM" ]]
    then
        echo "Different checksum found, compiling"
        echo "See $DIRECTORY/compilation.log"
        python setup.py install > compilation.log 2>&1
        COMPILATION_EXIT_CODE="$?"
        if [[ "$COMPILATION_EXIT_CODE" -ne "0" ]]
        then
            echo "Got $COMPILATION_EXIT_CODE exit code, finishing"
            exit "$COMPILATION_EXIT_CODE"
        fi
        SUM=`find . -type f ! -name "checksum.txt" -exec sha256sum {} \; | sha256sum`
        echo "$SUM" > checksum.txt
    fi
    popd > /dev/null
}

export MAX_JOBS="8"
compile_directory "csrc/ft_attention"
compile_directory "csrc/fused_dense_lib"
compile_directory "csrc/fused_softmax"
compile_directory "csrc/layer_norm"
compile_directory "csrc/layer_norm"
compile_directory "csrc/rotary"
compile_directory "csrc/xentropy"
compile_directory "."
echo "Running python tests"
pytest ./tests