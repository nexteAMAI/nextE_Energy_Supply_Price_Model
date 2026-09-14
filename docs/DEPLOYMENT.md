# DEPLOYMENT

## Live application

| Item | Value |
|---|---|
| URL | https://nexte-esb.streamlit.app |
| Host | Streamlit Community Cloud, account `nexteamai` |
| Source | GitHub `nexteAMAI/nextE_Energy_Supply_Price_Model`, branch `main`, main file `app.py` |
| Python | 3.12 (the CI matrix tests 3.11 and 3.12) |
| Dependencies | `requirements.txt` at the repository root (runtime only) |
| Configuration | `.streamlit/config.toml` (theme, upload limit 200 MB) - committed |
| Secrets | Streamlit Cloud, App settings, Secrets: section `[auth]` with `username` and `password`, typed by the administrator; never in the repository, the archive or the chat |
| Visibility | Public link on the free tier (the account's single private-app slot is held by the PMO tool); access is controlled by the in-app sign-in gate (`app/auth.py`) until the production access model is decided (OPEN_ITEMS DEPLOY) |
| Deployed | 14.09.2026, from commit `1dc2584` (Phase 5); the gate follows with the Phase 6 commit |

Every push to `main` redeploys the application automatically; the app reboots and re-installs
`requirements.txt` when it changes. The Reference Case fixtures in `data/reference/` ship with
the repository, so the application runs on the coded Reference Case without any upload.

## Sign-in gate

`app/auth.py` reads `st.secrets["auth"]["username"]` and `["password"]`; when the section is
absent the gate is off (local development) and the sidebar says so. Secrets shape:

```
[auth]
username = "<username>"
password = "<password>"
```

The comparison is constant-time; nothing typed is logged. A credential that has been sent over
a chat or an e-mail is to be rotated in the Secrets box (changes propagate in about a minute).

## Operating the deployment

- Reboot, logs, settings: share.streamlit.io, the app's menu (Manage app on the running app).
- Change the Python version or the URL: App settings, General.
- Restrict viewers to invited e-mails: App settings, Sharing - available when the private-app
  slot is free or on a paid plan.
- Roll back: re-point the app to a tag (App settings) or revert `main`.

## Local run (desk machine)

`Start_ESB_App.cmd` in the clone starts the application in the browser from `.venv`
(`pip install -e ".[dev,app]"` once). Without a local `.streamlit/secrets.toml` the gate is off.

## Release set (rule D-78: one atom)

A release is a tag `vX.Y.Z` on `main` carrying together: code, `config/parameters.yaml`, the
upload templates (`esb.importer.build_template`), the documentation set of docs/, and the
parity report of that version in the CEO folder `01_Tool/06_audits/`. The release pack copied
to `01_Tool/07_production/` holds the tagged source archive, the documents and the parity
report.
