# Team setup — interns work safely, each on their own Google Drive

Goal: your interns can develop the code **without affecting the original GitHub repo**, and
each one keeps the **dataset on their own Google Drive** (no shared Drive needed, no
re-fetching for everyone).

---

## 1. Protect the original repo (owner — one time)

On `github.com/prakhar443/illegal_mining`:

1. **Settings → Branches → Add branch protection rule** for `spearnet-colab` (and `main`
   if you use it):
   - ✅ Require a pull request before merging
   - ✅ Require approvals (1)
   - ✅ Do not allow bypassing the above
2. (Optional) **Settings → Collaborators**: you do **not** need to add interns as
   collaborators — forking works without it. Add them only if you prefer the branch model
   (Section 5).

Result: nothing reaches your repo except through a Pull Request that **you** review and
merge. Interns cannot push to your branches.

---

## 2. Each intern forks the repo (not clone-and-push)

On GitHub, each intern clicks **Fork** on `prakhar443/illegal_mining` → they get their own
`github.com/<intern>/illegal_mining`. All their commits live in **their fork**; your repo
is untouched.

---

## 3. Each intern's Colab points at their fork

In the notebook **cell 2**, the intern sets:

```python
REPO_URL = "https://github.com/<intern-username>/illegal_mining.git"
BRANCH   = "spearnet-colab"     # or their own working branch
```

Everything else in the notebook is unchanged.

### Pushing code changes from Colab (intern)

Colab needs a GitHub **Personal Access Token** (PAT) to push to a fork over HTTPS:

```python
# one-time per session, in a cell:
TOKEN = "ghp_xxx"      # the intern's own fine-grained PAT (repo: contents write on their fork)
!git -C illegal_mining remote set-url origin https://{TOKEN}@github.com/<intern>/illegal_mining.git
!git -C illegal_mining config user.email "intern@example.com"
!git -C illegal_mining config user.name  "intern"
!git -C illegal_mining add -A && git -C illegal_mining commit -m "my change" && git -C illegal_mining push
```

(Or, simpler for heavier coding: edit/commit on their **local machine** and just *run*
training in Colab.)

---

## 4. Contributing back to your repo

When an intern wants their work reviewed: GitHub → their fork → **Contribute → Open pull
request** into `prakhar443/illegal_mining`. You review and merge. The original only ever
changes via merges **you** approve.

### Keeping a fork up to date with your fixes

```bash
git remote add upstream https://github.com/prakhar443/illegal_mining.git
git fetch upstream
git merge upstream/spearnet-colab      # or: git rebase upstream/spearnet-colab
```

---

## 5. Dataset — each intern on their **own** Google Drive

The dataset is **per-account by design**: each intern mounts their own Drive, so chips and
the `spearnet_chips.zip` backup live on **their** Drive. Two ways to populate it:

### Option A (recommended) — fetch once, share a link, everyone restores

1. **You** (or one intern) fetch the full dataset once (notebook cells 5 → 6b, re-run 6b
   until done), then **6c** packages `spearnet_chips.zip`.
2. Upload that zip **once** to a public host:
   - **GitHub Release** asset on your repo (file ≤ 2 GB) — Releases → *Draft a new release*
     → attach the zip; **this does not bloat the repo** (releases are separate from code), and
   - or **Hugging Face dataset** (for > 2 GB).
3. Each intern sets `DATA_URL` in **cell 4c** to that link → it downloads to their Colab and
   (optionally) **6c** re-packages it to **their own** Drive. Now each intern has the dataset
   on their own Drive and **nobody re-fetches** from Planetary Computer.

### Option B — each intern fetches independently

Each intern just runs cells 5 → 6b → 6c on their account. The fetch is **resumable**
(re-run 6b after any crash) and lands on their own Drive. Slower (network-bound) but needs
no shared link.

Either way: **the original repo is never used to store data**, and each intern's dataset is
isolated on their own Drive.

---

## TL;DR

| Concern | Solution |
|---|---|
| Interns shouldn't touch your repo | They **fork**; you protect branches; merges via **PR review** |
| Interns push their code | To **their fork** (PAT), then open a PR |
| Dataset per intern | Each mounts **their own Drive**; restore from a shared **URL** (cell 4c) or fetch themselves |
| Don't store data on GitHub | Use a **Release asset** or **Hugging Face**, never repo files (100 MB/file cap) |
