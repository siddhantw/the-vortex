#!/bin/bash

# Zephyr Base URL
readonly jiraBaseUrl="https://jira.newfold.com"

# Check If All Required Parameters Are Provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <environment> <username> <password>"
    exit 1
fi

# Assigning Parameters To Variables
environment="$1"
username="$2"
userpassword="$3"
base64code=$(echo -n "$username:$userpassword" | base64)
userauth="Basic $base64code"
projectId="XXXXX"
versionId="-1"

# Function To Get Jira User Details
get_jira_user_details() {
    local result
    result=$(curl -s -H "Authorization: $userauth" -H "Content-Type: application/json" --request GET "$jiraBaseUrl/rest/api/latest/user?username=$username")
    echo "$result"
}

# Function To Get Cycle Details Or Create A New Cycle
get_or_create_cycle() {
    local cycleName cycleEnvironment getCycleDetailsUrl createCycleRequestUrl result cycleId

    # Determine Cycle Name And Environment
    case "$environment" in
        qamain)
            cycleName="Jarvis QA Main"
            cycleEnvironment="QA Main"
            ;;
        prod)
            cycleName="Jarvis Prod"
            cycleEnvironment="Prod"
            ;;
        stage)
            cycleName="Jarvis Stage"
            cycleEnvironment="Stage"
            ;;
        *)
            cycleName="Ad hoc"
            cycleEnvironment="Ad hoc"
            ;;
    esac

    getCycleDetailsUrl="$jiraBaseUrl/rest/zapi/latest/cycle?projectId=$projectId&versionId=$versionId"
    result=$(curl -s -H "Authorization: $userauth" -H "Content-Type: application/json" --request GET "$getCycleDetailsUrl")

    # Get Cycle ID If Exists
    cycleId=$(echo "$result" | jq -r --arg cycleName "$cycleName" '.[] | select(.name == $cycleName) | .id')

    # If Cycle Doesn't Exist, Create A New One
    if [ -z "$cycleId" ]; then
        createCycleRequestUrl="$jiraBaseUrl/rest/zapi/latest/cycle"
        result=$(curl -s -H "Authorization: $userauth" -H "Content-Type: application/json" \
            --request POST --data '{"name": "'"$cycleName"'", "projectId": "'"$projectId"'", "environment": "'"$cycleEnvironment"'", "versionId": "'"$versionId"'"}' "$createCycleRequestUrl")
        cycleId=$(echo "$result" | jq -r '.id')
    fi

    echo "$cycleId"
}

# Get Jira User Details
echo "Fetching Jira User Details..."
jiraUserDetails=$(get_jira_user_details)
jiraUserDisplayName=$(echo "$jiraUserDetails" | jq -r '.displayName')
jiraUserName=$(echo "$jiraUserDetails" | jq -r '.name')

# Get Or Create Cycle
echo "Getting or Creating Cycle Details..."
cycleId=$(get_or_create_cycle)

# Create A New Folder
folderName="$environment_Automation_$(date +'%Y-%m-%d %H:%M:%S')"
createFolderRequestUrl="$jiraBaseUrl/rest/zapi/latest/folder/create"
echo "Creating a folder..."
result=$(curl -s -H "Authorization: $userauth" -H "Content-Type: application/json" \
    --request POST --data '{"cycleId": "'"$cycleId"'", "name": "'"$folderName"'", "projectId": "'"$projectId"'", "versionId": "'"$versionId"'"}' \
    "$createFolderRequestUrl")
folderId=$(echo "$result" | jq -r '.id')

# Create An Automation Task
taskName="atom_$environment_$(date +'%Y-%m-%d %H:%M:%S')"
createAtomRequestUrl="$jiraBaseUrl/rest/zapi/latest/automation/job/create"
echo "Creating an A.T.O.M Task..."
result=$(curl -s -H "Authorization: $userauth" -H "Content-Type: application/json" \
    --request POST --data '{"projectId": "'"$projectId"'", "automationType": "UPLOAD", "taskName": "'"$taskName"'", "automationTool": "JUnit", "versionId": "'"$versionId"'", "assigneeDisplayName": "'"$jiraUserDisplayName"'", "assigneeName": "'"$jiraUserName"'", "cycleId": "'"$cycleId"'", "folderId": "'"$folderId"'"}' \
    "$createAtomRequestUrl")
automationJobId=$(echo "$result" | jq -r '.JOB_ID')

# Upload The XUnit File
xunitFilePath="@$(pwd)/mergedxunit.xml"
uploadRequestUrl="$jiraBaseUrl/rest/zapi/latest/automation/upload/$automationJobId"
echo "Uploading the xunit file to the task..."
result=$(curl -s -o headers -v -H "Authorization: $userauth" -H "Content-Type: multipart/form-data" -H "Content-Type: application/json" -F "file=$xunitFilePath" -XPOST "$uploadRequestUrl")
echo "$result"

# Execute The Zephyr A.T.O.M Task With The File Uploaded
executeRequestUrl="$jiraBaseUrl/rest/zapi/latest/automation/job/execute/$automationJobId"
echo "Executing Task..."
result=$(curl -s -H "Authorization: $userauth" -H "Content-Type: application/json" --data '{"executedBy": "'"$jiraUserName"'"}' \
      --request POST "$executeRequestUrl")
echo "$result"