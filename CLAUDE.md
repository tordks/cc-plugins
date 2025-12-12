# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Claude Code plugin marketplace repository containing personal plugins. It uses the marketplace format to distribute multiple plugins from a single repository.

## Architecture

```
cc-plugins/
├── .claude-plugin/
│   └── marketplace.json    # Marketplace manifest listing all plugins
└── plugins/
    └── <plugin-name>/
        ├── .claude-plugin/
        │   └── plugin.json # Plugin manifest
        └── skills/
            └── <skill-name>/
                ├── SKILL.md      # Main skill documentation (frontmatter + content)
                └── references/   # Supporting examples and documentation
```

**Marketplace manifest** (`.claude-plugin/marketplace.json`): Lists all plugins with name, description, version, and source path.

**Plugin manifest** (`plugins/<name>/.claude-plugin/plugin.json`): Defines individual plugin metadata.

**Skills**: Each skill has a `SKILL.md` with YAML frontmatter (name, description) followed by guidance content that Claude receives when the skill is triggered.

## Adding a New Plugin

1. Create `plugins/<plugin-name>/.claude-plugin/plugin.json`
2. Add skills under `plugins/<plugin-name>/skills/<skill-name>/SKILL.md`
3. Register the plugin in `.claude-plugin/marketplace.json`
