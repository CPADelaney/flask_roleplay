Below are a few small, high‑impact patches to bring ESLint into the repo, enforce consistent style, and get CI to run your linter automatically.

1) Add an ESLint config  
2) Ignore build/output dirs  
3) Wire up `npm run lint` in your package.json  
4) Add a GitHub Action to run the linter on every PR/push  

```diff
diff --git a/.eslintrc.js b/.eslintrc.js
new file mode 100644
index 0000000..abcdef1
--- /dev/null
+++ b/.eslintrc.js
@@ -0,0 +1,25 @@
+module.exports = {
+  root: true,
+  env: {
+    node: true,
+    browser: true,
+    es6: true,
+  },
+  parser: '@typescript-eslint/parser',
+  parserOptions: {
+    ecmaVersion: 2020,
+    sourceType: 'module',
+  },
+  extends: [
+    'eslint:recommended',
+    'plugin:@typescript-eslint/recommended'
+  ],
+  plugins: ['@typescript-eslint'],
+  rules: {
+    // stylistic rules
+    'indent': ['error', 2, { SwitchCase: 1 }],
+    'quotes': ['error', 'single', { avoidEscape: true }],
+    'semi': ['error', 'always'],
+    // catch commonbugs
+    'no-console': 'warn',
+    '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
+  },
+};
diff --git a/.eslintignore b/.eslintignore
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/.eslintignore
@@ -0,0 +1,3 @@
+node_modules/
+dist/
+coverage/
diff --git a/package.json b/package.json
index a1b2c3d..d4e5f6g 100644
--- a/package.json
+++ b/package.json
@@ -10,7 +10,8 @@
   "scripts": {
     "start": "node index.js",
-    "test": "echo \"Error: no test specified\" && exit 1"
+    "test": "echo \"Error: no test specified\" && exit 1",
+    "lint": "eslint . --ext .js,.ts"
   },
   "dependencies": {
     /* … */
@@ -25,6 +26,9 @@
   },
   "devDependencies": {
     /* existing devDeps… */
+    "eslint": "^8.0.0",
+    "@typescript-eslint/parser": "^5.0.0",
+    "@typescript-eslint/eslint-plugin": "^5.0.0"
   }
 }
diff --git a/.github/workflows/lint.yml b/.github/workflows/lint.yml
new file mode 100644
index 0000000..fedcba9
--- /dev/null
+++ b/.github/workflows/lint.yml
@@ -0,0 +1,26 @@
+name: Lint
+
+on:
+  push:
+    branches: [ main ]
+  pull_request:
+    branches: [ main ]
+
+jobs:
+  eslint:
+    name: ESLint
+    runs-on: ubuntu-latest
+    steps:
+      - uses: actions/checkout@v3
+      - name: Use Node.js
+        uses: actions/setup-node@v3
+        with:
+          node-version: '16'
+      - name: Install dependencies
+        run: npm ci
+      - name: Run ESLint
+        run: npm run lint
```

Apply these four diffs to:

- enforce a consistent style,
- catch unused vars/console calls,
- provide a `npm run lint` entrypoint, and
- have CI guard your branches.

Let me know if you have any other specific lint errors you’d like addressed!