# ASTAA Store Audit Workflow

Mistral Workflows project that audits the ASTAA Shopify store checkout funnel
through the `astaa_7e0d` UCP/MCP connector.

## Layout

- `apps/worker/src/worker/workflows/astaa_audit.py` - the `astaa-store-audit` workflow
- `apps/worker/Dockerfile.worker` - worker image for managed deployment (build dir: `apps/worker`)

## Local test

```bash
uv sync
uv run pytest -q
```

The audit workflow uses the triggering user's connector credentials
(`on_behalf_of=True`, hardened deployment). It never completes a checkout.
