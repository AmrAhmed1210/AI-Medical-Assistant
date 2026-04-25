# AI-Medical-Assistant

**AI-Medical-Assistant** is a comprehensive platform designed to enhance healthcare delivery by combining Artificial Intelligence with user-friendly interfaces. The system allows medical professionals to train and deploy natural language models for analyzing patient data, automatically generate structured reports, and provide conversational support through chatbot features. It includes:

- A Python-based machine learning core for model training and inference.
- A responsive web application for administrators, doctors, and staff.
- A mobile app for patients and healthcare providers built with React Native.

Together, these components create an ecosystem that streamlines documentation, improves communication, and brings intelligent insights directly into clinical workflows.

## Repository Structure

- **`scr/` & `Lora/`** – Core AI model logic, training and inference scripts:
  - `train.py`, `tarin.py` (training pipelines)
  - `preprocess.py` (data preparation)
  - `inference.py` (model serving)
  - Output and experimentation files under `results/` and `Notebook/`.

- **`web/`** – React + Vite web application:
  - Components, pages, layouts, and styles in `src/`, `api/`, and `styles/`.
  - Admin dashboard, user management, and utility modules.
  - Build configuration: `package.json`, `vite.config.js`, ESLint.

- **`DoctorApp/`** – Mobile application (React Native & Expo):
  - Screens for doctors and patients (authentication, chat, scheduling, AI reports, profiles).
  - Organized assets, components, and type definitions.

- **Supporting files**
  - `requirements.txt` for Python dependencies.
  - Misc scripts, reports, and notebooks across the workspace.

## Getting Started

1. **Python environment:**
   ```bash
   pip install -r requirements.txt
   ```
   Run training or inference via scripts in `scr/`.

2. **Web app:**
   ```bash
   cd web
   npm install
   npm run dev
   ```

3. **Mobile app:**
   ```bash
   cd DoctorApp
   npm install
   expo start
   ```

## Development Notes

- Ensure AI models are trained and saved before attempting inference.
- The web and mobile UIs currently depend on local or cloud endpoints for model predictions.
- Stylesheets are located in `web/styles` and can be adjusted per design requirements.

## Next Steps

- Complete and validate the ML pipeline to produce deployable models.
- Integrate inference endpoints with front-end applications.
- Polish UI/UX and address build/runtime issues in both web and mobile projects.
- Prepare deployment configurations for backend services or cloud hosting.

---

_Last updated: March 4, 2026_

# Admin Credentials:
- Email: [admin@medbook.com]
- Password: [123456789]