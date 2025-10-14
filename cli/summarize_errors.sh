#!/bin/bash

# Exit if SLURM job ID isn't provided as a command line argument
if [ -z "$1" ]; then
    echo "Usage: $0 <job_id>"
    echo "Error: No job_id argument provided."
    exit 1
fi

# Enable nullglob so unmatched globs expand to nothing
shopt -s nullglob  

job_id=$1
LOGS_DIR="./output/logs"

# Check for ./output/logs
if [[ ! -d "$LOGS_DIR" ]]; then
    echo "No logs directory found at ./$LOGS_DIR"
    exit 1
fi

# Check for subdirectories under ./output/logs
subdirs=("$LOGS_DIR"/*)
if (( ${#subdirs[@]} == 0 )); then
    echo "No job name subdirectory found under ./$LOGS_DIR"
    exit 1
fi

# Check for ./output/logs/*/err 
err_dirs=("$LOGS_DIR"/*/err)
if (( ${#err_dirs[@]} == 0 )); then
    echo "No error directory found under ./$LOGS_DIR/*"
    exit 1
fi

# Check for ./output/logs/*/err/job_id 
job_dirs=("$LOGS_DIR"/*/err/"$job_id")
if (( ${#job_dirs[@]} == 0 )); then
    echo "No directory for job ${job_id} found under $LOGS_DIR/*/err/"
    exit 1
fi

# Check for .err files under ./output/logs/*/err/job_id 
err_files=("$LOGS_DIR"/*/err/"$job_id"/*.err)
if (( ${#err_files[@]} == 0 )); then
    echo "No .err files found under ./$LOGS_DIR/*/err/$job_id"
    exit 1
fi

num_err_files=${#err_files[@]}

printf '%*s\n' 80 '' | tr ' ' '-'
echo "Found $num_err_files .err files for job $job_id. Reading them now..."

# Create dump directory
dump="./output/dump"
mkdir -p "$dump"
output_file="$dump/${job_id}.csv"

# Write CSV header
csv_header="job_name,array_task_id,has_error,has_done_file,error_message"
echo "$csv_header" > "$output_file"

num_files_read=0
num_failed_jobs=0
job_name=""
error_files=()
failed_task_ids=()

# Get all task IDs from err files
all_task_ids=()
for err_file in "${err_files[@]}"; do
    [ -e "$err_file" ] || continue
    task_id=$(basename "$err_file" .err)
    all_task_ids+=("$task_id")
done

# Sort task IDs numerically
IFS=$'\n' all_task_ids=($(sort -n <<<"${all_task_ids[*]}"))
unset IFS

# Get job name from first err file
if [ ${#err_files[@]} -gt 0 ]; then
    job_name=$(echo "${err_files[0]}" | cut -d'/' -f4)
fi

# Check done directory
done_dir="${LOGS_DIR}/${job_name}/done/${job_id}"
missing_done_files=()
num_missing_done=0

for err_file in "${err_files[@]}"; do
    # Skip file if it doesn't exist
    [ -e "$err_file" ] || continue

    num_files_read=$((num_files_read + 1))

    # Parse job_name from ./output/logs/<job_name>/err/<job_id>/<task_id>.err
    job_name=$(echo "$err_file" | cut -d'/' -f4)
    
    # task_id is the file name without the .err extension
    task_id=$(basename "$err_file" .err)

    has_error="no"
    has_done_file="no"
    error_message=""

    # Check for errors
    if [ -s "$err_file" ]; then
        if grep -q "Traceback" "$err_file"; then
            has_error="yes"
            num_failed_jobs=$((num_failed_jobs + 1))
            error_files+=("$err_file")
            failed_task_ids+=("$task_id")

            # Extract first line of the traceback block for preview
            error_message=$(awk '/Traceback/ {f=1} f' "$err_file" | tr -d '\000-\011\013\014\016-\037' | tr '\n' '␤' | sed 's/"/""/g')
        fi
    fi

    # Check for corresponding done file
    done_file="${done_dir}/${task_id}.done"
    if [ -f "$done_file" ]; then
        has_done_file="yes"
    else
        has_done_file="no"
        missing_done_files+=("$task_id")
        num_missing_done=$((num_missing_done + 1))
    fi

    # Save row to CSV file
    printf '%s,%s,%s,%s,"%s"\n' "$job_name" "$task_id" "$has_error" "$has_done_file" "$error_message" >> "$output_file"
done

echo "Finished reading ${num_err_files} .err files"
printf '%*s\n' 80 '' | tr ' ' '-'

if [ -d "$done_dir" ]; then
    num_done_files=$(find "$done_dir" -maxdepth 1 -type f -name "*.done" | wc -l)
    echo "Found $num_done_files .done files for job ${job_id}"
else
    echo "Directory $done_dir does not exist!"
    num_missing_done=${#all_task_ids[@]}
    missing_done_files=("${all_task_ids[@]}")
fi

printf '%.0s-' {1..80}
echo  # To add a newline

# Report missing done files
if [ "$num_missing_done" -gt 0 ]; then
    echo "${num_missing_done} task(s) do not have done files. Tasks without done files:"
    
    # Sort missing task IDs numerically
    IFS=$'\n' sorted_missing=($(sort -n <<<"${missing_done_files[*]}"))
    unset IFS
    
    # Print each missing task with its error file path
    for task_id in "${sorted_missing[@]}"; do
        # Construct the error file path for this task
        err_file_path="${LOGS_DIR}/${job_name}/err/${job_id}/${task_id}.err"
        echo "- Task ${task_id}: ${err_file_path}"
    done
    
    echo ""
    # Create array format for resubmission
    if [ "${#sorted_missing[@]}" -gt 0 ]; then
        IFS=','  
        echo "To resubmit missing tasks: --array=${sorted_missing[*]}"
        unset IFS  
    fi
else 
    echo "All $num_err_files tasks have done files"
fi
echo ""