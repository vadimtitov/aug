# Portainer Tools Setup

## Prerequisites

A running Portainer instance accessible from the AUG container.

## 1. Get your API token

1. Log into Portainer → click your username (top right) → **Account Settings**
2. Scroll to **Access tokens** → **Add access token**
3. Give it a name (e.g. `aug`) → copy the token immediately (shown once)

## 2. Get your endpoint ID

1. Portainer **Home** → click your environment
2. Check the browser URL: `http://your-portainer/#/endpoints/2/docker/...`
3. The number after `/endpoints/` is your endpoint ID

## 3. Configure

Add to your `.env`:

```env
PORTAINER_URL=http://portainer:9000      # or http://<host>:<port>
PORTAINER_API_TOKEN=ptr_xxxxxxxxxxxx
PORTAINER_ENDPOINT_ID=2                  # from step 2, default is 1
```

## Available tools

| Tool | What it does |
|------|-------------|
| `portainer_list_containers` | List all containers with status |
| `portainer_container_logs` | Get recent logs from a container |
| `portainer_container_action` | Start / stop / restart / remove a container |
| `portainer_list_stacks` | List all stacks |
| `portainer_deploy_stack` | Deploy or update a stack from a compose file |
| `portainer_stack_action` | Start / stop / remove a stack |

All tools are disabled (return a clear message) if `PORTAINER_URL` or `PORTAINER_API_TOKEN` is not set.
