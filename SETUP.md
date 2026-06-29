# Setup — one time only

Run these commands in **Git Bash** (from inside the `index_html_tracker` folder):

```bash
# 1. Initialise git and push to a new GitHub repo
cd ~/Desktop/index_html_tracker      # adjust path to wherever you moved this folder

git init
git add .
git commit -m "Initial commit — Ekaiva NSE Index HTML tracker"

# Create the repo on GitHub and push  (gh CLI must be installed and logged in)
gh repo create ekaivawealth/ekaiva-index-html --public --source=. --remote=origin --push
```

### 2. Enable GitHub Pages

After pushing:
1. Go to `https://github.com/ekaivawealth/ekaiva-index-html`
2. **Settings → Pages**
3. Under "Build and deployment" → Source: **Deploy from a branch**
4. Branch: **main** | Folder: **/ (root)**
5. Click **Save**

Your public URL will be: **https://ekaivawealth.github.io/ekaiva-index-html/**

### 3. Trigger the first run

Go to `https://github.com/ekaivawealth/ekaiva-index-html/actions`
→ Click **update-dashboard** → **Run workflow** → **Run workflow**

After ~2 minutes the full dashboard will be live.

### 4. After that — fully automatic

The workflow runs every **weekday at 20:00 IST** (14:30 UTC).
It reads seed history from `ekaiva-tracker` (your existing repo) and fetches
today's live prices from Yahoo Finance / NiftyIndices — then rewrites `index.html`.

No secrets needed. No manual steps.
