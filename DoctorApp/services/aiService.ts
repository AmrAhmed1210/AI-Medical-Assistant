import axios from "axios";

// For real devices: use your computer's IP (e.g. 192.168.1.5)
// For Android Emulator: use 10.0.2.2
const AI_IP = "192.168.1.4"; // Change this if your computer IP changes
const AI_SERVER_URL = `http://${AI_IP}:8000`; 

export const analyzePatientHistory = async (history: any) => {
    try {
        const response = await axios.post(`${AI_SERVER_URL}/analyze-history`, history);
        return response.data; // Return the whole object {analysis_en, analysis_ar}
    } catch (error) {
        console.error("AI Analysis failed:", error);
        throw error;
    }
};

export const summarizeSurgery = async (description: string) => {
    try {
        const response = await axios.post(`${AI_SERVER_URL}/summarize-surgery`, { description });
        return response.data; // { summary_en, summary_ar }
    } catch (error) {
        console.error("Surgery summary failed:", error);
        throw error;
    }
};

export const summarizeMedicalItem = async (type: string, description: string) => {
    try {
        const response = await axios.post(`${AI_SERVER_URL}/summarize-medical-item`, { type, description });
        return response.data; // { summary_en, summary_ar }
    } catch (error) {
        console.error("Medical item summarization failed:", error);
        return { summary_en: description, summary_ar: description };
    }
};

export const analyzeVitalsAdvice = async (vitals: any[], patientInfo: any) => {
    try {
        const response = await axios.post(`${AI_SERVER_URL}/analyze-vitals`, { vitals, patient_info: patientInfo });
        return response.data; // {advice_en, advice_ar}
    } catch (error) {
        console.error("Vitals analysis failed:", error);
        return { advice_en: "Keep monitoring.", advice_ar: "استمر في المتابعة." };
    }
};

export const checkMedicationSafety = async (medication: string, history: any) => {
    try {
        const response = await axios.post(`${AI_SERVER_URL}/check-medication-safety`, { medication, history });
        return response.data; // {safety_en, safety_ar}
    } catch (error) {
        console.error("Medication safety check failed:", error);
        return { safety_en: "Consult doctor.", safety_ar: "استشر طبيبك." };
    }
};

export const analyzeMedicalImage = async (uri: string, type: "prescription" | "lab" = "prescription", patientContext: string = "") => {
    try {
        const formData = new FormData();
        const filename = uri.split("/").pop() || "image.jpg";
        const match = /\.(\w+)$/.exec(filename);
        const mimeType = match ? `image/${match[1]}` : "image/jpeg";

        formData.append("file", { uri, name: filename, type: mimeType } as any);
        formData.append("type", type);
        if (patientContext) {
            formData.append("patient_context", patientContext);
        }
        
        const response = await axios.post(`${AI_SERVER_URL}/analyze-image`, formData, {
            headers: {
                "Content-Type": "multipart/form-data",
            },
        });
        return response.data;
    } catch (error) {
        console.error("Image analysis failed:", error);
        throw error;
    }
};
