#!/bin/bash

# Compute overall diffs
MODIFIED_FILES=`git status -s | awk '{if (\$1 == "M") print \$2}'`

MODIFIED_COUNT="$(echo "${MODIFIED_FILES}" | wc -l)"
if [ -z "${MODIFIED_COUNT}" ]; then
    echo "<< no change >>"
    exit
fi

echo
echo -e "\e[93m${MODIFIED_FILES}\e[0m"
echo

# Compute diffs one by one
MODIFIED_ID=1
for MODIFIED_FILE in ${MODIFIED_FILES}
do
    # Get names
    GIT_NAME=`git ls-files --full-name ${MODIFIED_FILE}`

    DISPLAY_NAME=${GIT_NAME:-$MODIFIED_FILE}
    echo -e "==== [${MODIFIED_ID}/${MODIFIED_COUNT}] \e[93m${DISPLAY_NAME}\e[0m ===="
    MODIFIED_ID=$(expr ${MODIFIED_ID} + 1)

    if [ -z "${GIT_NAME}" ]; then
        echo "Warning: unable to get full name for ${MODIFIED_FILE}"
        continue
    fi

    # Get git content
    GIT_FILE=git.`echo ${MODIFIED_FILE} | sed "s#/#-#g"`.tmp
    rm -f $GIT_FILE
    git show master:$GIT_NAME > $GIT_FILE

    # Get diff with local file
    aDiff="$(diff -E -b -w -B -a $GIT_FILE $MODIFIED_FILE)"
    aDiffSize="$(echo "${aDiff}" | wc -l)"
    if [ -z "${aDiffSize}" ]; then
        echo "<< no change >>"
    elif [ $aDiffSize -ge 40 ]; then
        echo "<< big diff of ${aDiffSize} lines >>"
    else
        python ~/cdiff.py $GIT_FILE $MODIFIED_FILE
    fi

    # Ask user for action
    read -p "next? [Enter/v/d/q]: " ANSWER
    case $ANSWER in
        v|V)
            vimdiff $GIT_FILE $MODIFIED_FILE
            rm -f $GIT_FILE
            continue
            ;;
        q|Q)
            rm -f $GIT_FILE
            exit
            ;;
        d|D)
            python ~/cdiff.py $GIT_FILE $MODIFIED_FILE
            rm -f $GIT_FILE
            continue
            ;;
        *)
    esac

    rm -f $GIT_FILE
done
