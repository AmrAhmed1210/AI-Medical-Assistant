import axios from "axios";

// This is the address of your Python AI server
// Using local IP 192.168.43.216 to allow real devices on the same WiFi to connect
const AI_SERVER_URL = "http://192.168.43.216:8000"; 

export const analyzePatientHistory = async (history: any) => {
    try {
        const response = await axios.post(`${AI_SERVER_URL}/analyze-history`, history);
        return response.data.analysis;
    } catch (error) {
        console.error("AI Analysis failed:", error);
        throw error;
    }
};

export const summarizeSurgery = async (description: string) => {
    try {
        const response = await axios.post(`${AI_SERVER_URL}/summarize-surgery`, { description });
        return response.data.summary;
    } catch (error) {
        console.error("Surgery summary failed:", error);
        throw error;
    }
};

export const summarizeMedicalItem = async (type: string, description: string) => {
    try {
        const response = await axios.post(`${AI_SERVER_URL}/summarize-medical-item`, { type, description });
        return response.data.refined_text;
    } catch (error) {
        console.error("Medical item summarization failed:", error);
        return description;
    }
};

export const analyzeVitalsAdvice = async (vitals: any[], patientInfo: any) => {
    try {
        const response = await axios.post(`${AI_SERVER_URL}/analyze-vitals`, { vitals, patient_info: patientInfo });
        return response.data.advice;
    } catch (error) {
        console.error("Vitals analysis failed:", error);
        return "Keep monitoring your vitals.";
    }
};

export const checkMedicationSafety = async (medication: string, history: any) => {
    try {
        const response = await axios.post(`${AI_SERVER_URL}/check-medication-safety`, { medication, history });
        return response.data.safety_report;
    } catch (error) {
        console.error("Medication safety check failed:", error);
        return "Consult your doctor for safety.";
    }
};

export const analyzeMedicalImage = async (uri: string, type: "prescription" | "lab" = "prescription") => {
    try {
        const formData = new FormData();
        const filename = uri.split("/").pop() || "image.jpg";
        const match = /\.(\w+)$/.exec(filename);
        const mimeType = match ? `image/${match[1]}` : "image/jpeg";

        formData.append("file", { uri, name: filename, type: mimeType } as any);
        
        const response = await axios.post(`${AI_SERVER_URL}/analyze-image?type=${type}`, formData, {
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
