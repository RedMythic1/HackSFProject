# Tech Deep Dive Web Application

This directory contains the web application for Tech Deep Dive, which provides an interface for exploring technology articles with AI-generated questions and answers.

## Project Structure

- `server.py` - Backend Flask server that handles article caching, question generation, and serving content
- `src/` - Frontend TypeScript source code
- `public/` - Static frontend assets
- `dist/` - Compiled frontend code
- `.cache/` - Directory for cached articles and generated content
- `manage.sh` - Main management script for all operations

## Getting Started

All operations have been consolidated into a single management script: `manage.sh`

To use the script:

```bash
./manage.sh [command]
```

Available commands:

- `setup` - Set up the environment (copy ansys.py, create cache directories)
- `start` - Start both the server and frontend
- `stop` - Stop running server and frontend processes
- `cache` - Cache articles from Hacker News
- `questions` - Generate questions for cached articles
- `status` - Check the status of the cache and server
- `help` - Show help message

## Typical Workflow

1. **Setup**: `./manage.sh setup`
2. **Start services**: `./manage.sh start`
3. **Cache articles**: `./manage.sh cache`
4. **Generate questions**: `./manage.sh questions`
5. **Check status**: `./manage.sh status`
6. **Stop services**: `./manage.sh stop`

## Development

The frontend is built with TypeScript and compiled with Webpack. The backend is a Flask server that handles API requests.

To start development:

1. Run `./manage.sh start` to start both services
2. Make changes to the source code
3. For frontend changes, webpack will automatically recompile
4. For backend changes, restart the server: `./manage.sh stop` then `./manage.sh start`

## Troubleshooting

- Check log files: `server.log` and `frontend.log`
- Run `./manage.sh status` to check system status
- Make sure `ansys.py` is properly set up (use `./manage.sh setup` to verify) 