# cc-plugins

Personal collection of Claude Code plugins.

## Installation

```bash
# Add from local
/plugin marketplace add /path/to/cc-plugins

# Or from GitHub:
/plugin marketplace add tordks/cc-plugins

# Install a specific plugin
/plugin install image-analysis@cc-plugins
```

## Available Plugins

| Plugin | Description |
|--------|-------------|
| [image-analysis](./plugins/image-analysis) | Image and video segmentation tools |

## Structure

```
cc-plugins/
├── .claude-plugin/
│   └── marketplace.json
├── plugins/
│   └── image-analysis/
│       ├── .claude-plugin/
│       │   └── plugin.json
│       └── skills/
│           └── sam3/
└── README.md
```
