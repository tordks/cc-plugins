# claude-plugins

Personal collection of Claude Code plugins.

## Installation

```bash
# Add this marketplace
/plugin marketplace add /path/to/claude-plugins
# Or from GitHub:
/plugin marketplace add tordks/claude-plugins

# Install a specific plugin
/plugin install image-analysis@claude-plugins
```

## Available Plugins

| Plugin | Description |
|--------|-------------|
| [image-analysis](./plugins/image-analysis) | Image and video segmentation tools |

## Structure

```
claude-plugins/
├── .claude-plugin/
│   └── marketplace.json       # Marketplace manifest
├── plugins/
│   └── image-analysis/        # Image segmentation plugin
│       ├── .claude-plugin/
│       │   └── plugin.json
│       └── skills/
│           └── sam3/          # SAM3 API guidance
└── README.md
```
