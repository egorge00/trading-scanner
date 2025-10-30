# 📘 Spécification fonctionnelle – MVP Scanner d’Opportunités

## 🎯 Objectif
Créer un outil personnel d’analyse et de détection d’opportunités d’investissement (actions Europe + S&P500) :
- Accessible depuis le **web (mobile & desktop)**.
- Protégé par **authentification**.
- Capable de **scanner** automatiquement les marchés chaque jour.
- D’**analyser** les valeurs via des indicateurs techniques.
- Et de **t’envoyer un rapport e-mail quotidien** à 09h CET.

---

## 🧩 1. Architecture

| Composant | Description |
|------------|--------------|
| **Front-end** | Application **Streamlit Cloud** (UI responsive, login interne) |
| **Back-end** | Scripts Python embarqués (analyse, scoring, mail) |
| **Données de marché** | `yfinance` (actions européennes & S&P500) |
| **Stockage** | Pour le MVP : fichiers CSV/Google Sheet ; plus tard PostgreSQL |
| **Envoi d’e-mails** | SMTP Gmail (mot de passe d’application) |
| **Planification** | GitHub Actions (fetch + mail à 09:00 CET) |
| **Authentification** | Simple via bcrypt (login `egorge`) |
| **Accès** | Public Streamlit URL, mot de passe requis |

---

## 📊 2. Fonctionnalités principales

### 🔹 Univers de titres
- Ensemble initial :
  - Actions **Europe** (Euronext, Deutsche Börse, etc.)
  - **S&P 500**
- Possibilité d’**ajouter ou supprimer** manuellement des ISIN/tickers via l’interface.

### 🔹 Indicateurs / KPI calculés
| Catégorie | Indicateur | Utilisation |
|------------|-------------|--------------|
| **Momentum** | RSI, MACD, variation 5j/20j | Détection sur/sous-achat |
| **Tendance** | MM20, MM50, MM200, croisement haussier/baissier | Direction du marché |
| **Volatilité** | Bollinger bands, ATR | Filtrer les titres trop volatils |
| **Position relative** | % vs plus haut/bas 52 semaines | Potentiel de rattrapage |
| **Événements** | earnings, dividendes, news sentiment | Ajustement score |
| **Score global** | pondération de tous les signaux | Génère le code couleur 🟢⚪🟠🔴 |

### 🔹 Score d’opportunité
| Couleur | Interprétation | Action suggérée |
|----------|----------------|-----------------|
| 🟢 | Opportunité forte | Achat / Renforcement |
| ⚪ | Neutre | Attente |
| 🟠 | À surveiller | Alléger |
| 🔴 | Signal de sortie | Vente |

---

## 💼 3. Gestion manuelle

### 🔹 Watchlist
- Liste des valeurs suivies (`ISIN`, `nom`, `ticker`, `marché`).
- Ajout/suppression via UI.
- Export/import CSV.

### 🔹 Positions en cours
- Indiquer manuellement :
  - ISIN
  - Date d’entrée
  - Note libre (optionnelle)
- L’app affiche le **score actuel** de chaque position.
- Envoi d’alerte immédiate si score ≤ -2.

---

## 🕘 4. Planification & e-mails

### 🧭 Tâches automatiques
| Heure CET | Tâche | Description |
|------------|--------|-------------|
| 07:30 | Fetch & analyse | Téléchargement des clôtures de la veille + calculs |
| 09:00 | Rapport e-mail | Envoi du mail complet |
| (24/7) | Alertes instantanées | Mail si une position passe 🔴 (cooldown 6h) |

### ✉️ Rapport e-mail (HTML)
Sections :
1. **Top 10 opportunités haussières 🟢**
2. **Positions en cours (avec score)**
3. **Alertes ventes 🔴**
4. **Événements à venir (earnings, dividendes)**

---

## 🔒 5. Sécurité & accès
- Authentification locale avec bcrypt (`egorge` / ton mot de passe).
- Page login Streamlit (compatible trousseau iOS).
- Accès HTTPS via URL Streamlit Cloud.
- Aucun stockage de mot de passe en clair.

---

## 🌐 6. Roadmap future
| Étape | Objectif |
|--------|-----------|
| **v1.0 (MVP actuel)** | Streamlit Cloud + e-mail quotidien |
| **v1.1** | Ajout des tickers via interface + Google Sheet persistant |
| **v1.2** | Passage à base PostgreSQL hébergée (Neon ou Supabase) |
| **v1.3** | Module de PnL + positions historiques |
| **v1.4** | News & sentiment AI (résumés automatiques) |
