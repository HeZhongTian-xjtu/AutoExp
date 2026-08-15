# Security Policy

AutoExp executes generated and repaired machine-learning code. Treat this repository as a local demonstration unless the deployment is independently hardened.

## Reporting a vulnerability

Please do not disclose credentials, exploit details, private datasets, or vulnerable source code in a public Issue. Use GitHub Security Advisories or contact the repository maintainers through the private channel associated with the repository.

Include a concise description, affected version or commit, reproduction steps that do not contain secrets, and the potential impact. The maintainers will acknowledge a valid report and coordinate a fix or mitigation.

## Security boundaries

- Keep API keys in `.env` or a secret manager; never commit them.
- Use `DockerExecutor` for generated or repaired code. `LocalExecutor` is for trusted development.
- Review source and logs before allowing them to leave the machine for an external LLM provider.
- Keep Redis, MLflow, and the API private unless authentication, TLS, quotas, and network policy are configured.
- Rotate any key that has appeared in logs, screenshots, chats, or issue discussions.
