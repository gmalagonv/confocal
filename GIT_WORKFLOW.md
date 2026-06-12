# Git Workflow

## Branches
- `main` — stable shared version (merge here when ready to share)
- `gerardo-branch` — Gerardo's active work
- `carmina-brach` — Carmina's active work

Branches stay alive — they are personal workspaces. Neither is deleted automatically.

---

## Daily workflow (each person)

```bash
# 1. Before starting work — get latest from main
git checkout main
git pull github main

# 2. Switch to your branch
git checkout gerardo-branch   # or carmina-brach for Carmina

# 2b. Bring any new changes from main into your branch
git merge main

# 3. Work normally, then stage and commit
git add analysis_notebooks/26_06_05_gerardo.ipynb  # specific files
git commit -m "description of what you did"

# 4. Push your branch to GitHub
git push github gerardo-branch   # or carmina-brach for Carmina
```

---

## When you want to share/merge with each other

Whoever is ready merges into `main`:

```bash
git checkout main
git pull github main          # get any changes the other person pushed
git merge gerardo-branch      # merge your work in
git push github main          # push merged main
```

Then the other person syncs:

```bash
git checkout main
git pull github main          # get the update
git merge main                # bring it into their branch (keeps branches current)
```

---

**Key habit:** always `git pull github main` before merging into `main`, so you don't overwrite the other person's work.
