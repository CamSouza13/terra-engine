# Deploying the Terra platform

The platform is a single long-running Python process (engine loop + web console +
JSON API) that keeps all state in `$TERRA_HOME`. Any host that gives you a
persistent disk and a always-on process works; the repo ships a Fly.io config
because it's the cleanest fit.

## Fly.io (recommended)

One-time setup:

```
fly launch --no-deploy --copy-config --name terra-platform
fly volumes create terra_data --size 1 --region iad
fly deploy
```

`fly.toml` already mounts the `terra_data` volume at `/data`, sets
`TERRA_HOME=/data` and `TERRA_AUTH=1`, keeps one machine always running (the
engine loop must not be stopped), and forces HTTPS. After `fly deploy` your
console is live at `https://terra-platform.fly.dev` and the pricing page at
`/pricing`.

Point a custom domain at it:

```
fly certs add console.terralaboratories.com
```

## Secrets (never commit these)

Set them with `fly secrets set KEY=value` — they land in the machine environment
without touching git:

```
fly secrets set \
  TERRA_SMTP_HOST=smtp.postmarkapp.com \
  TERRA_SMTP_PORT=587 \
  TERRA_SMTP_USER=... \
  TERRA_SMTP_PASS=... \
  TERRA_SMTP_FROM=alerts@terralaboratories.com
```

Email alerts send only when `TERRA_SMTP_*` is set; Slack and webhook alerts need
no server config (the destination URL lives on the rule).

## Stripe billing

The paywall enforces plans locally; to actually charge, configure Stripe:

```
fly secrets set \
  STRIPE_SECRET_KEY=sk_live_... \
  STRIPE_PRICE_PRO=price_... \
  STRIPE_PRICE_FLEET=price_... \
  STRIPE_WEBHOOK_SECRET=whsec_...
```

Create the Pro (and optional Fleet) recurring **prices** in the Stripe dashboard
and paste their `price_…` IDs above. Then add a webhook endpoint pointing at
`https://<your-host>/api/billing/webhook` and subscribe it to
`checkout.session.completed` and `customer.subscription.updated/deleted`.

With those set, the console's Upgrade button opens a real Stripe Checkout session;
after payment Stripe calls the webhook and the workspace plan flips automatically.
Without them, upgrade falls back to a documented stub that sets the plan directly
(useful for demos and local testing).

## Connecting a node

From the deployed console, open **Settings -> Enroll a node** to mint a one-time
token, then on the node hardware:

```
terra node --enroll et_XXXX --server https://terra-platform.fly.dev --name pond-a --domain aquaculture
```

The node redeems the token, stores its own key under `$TERRA_HOME/node_creds.json`,
and reports a heartbeat every 10s (`--interval` to change). It shows up live in the
**Fleet** view. Reporting is best-effort: if the platform is unreachable the node
keeps running locally and resumes reporting when it returns.

## Data & backups

Everything durable — accounts, sessions, workspaces, plans, alert rules and
events, node registry, API keys, engine config, and estimate history — lives under
`/data` (SQLite `terra.db` plus `data/history.jsonl`). Snapshot the Fly volume, or
for real scale swap SQLite for hosted Postgres behind the same `accounts`/`registry`
functions.
