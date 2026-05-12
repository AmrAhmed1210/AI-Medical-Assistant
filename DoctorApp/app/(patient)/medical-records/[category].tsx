import React, { useEffect, useState, useCallback } from "react";
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  StatusBar, ActivityIndicator, TextInput, Modal, Alert,
} from "react-native";
import { useRouter, useLocalSearchParams } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { Sparkles } from "lucide-react-native";
import { LinearGradient } from "expo-linear-gradient";
import { COLORS } from "../../../constants/colors";
import { getMyPatientId } from "../../../services/authService";
import {
  getAllergies, createAllergy, deleteAllergy,
  getChronicDiseases, deleteChronicDisease,
  getMedications, deleteMedication,
  getVitals, deleteVital,
  getSurgeries, createSurgery, deleteSurgery,
  getPatientDocuments, uploadPatientDocument, deletePatientDocument,
  createChronicDisease,
  AllergyRecord, ChronicDisease, Medication, VitalReading, SurgeryRecord, PatientDocument,
} from "../../../services/medicalRecordService";
import Toast from "react-native-toast-message";
import { useLanguage } from "../../../context/LanguageContext";
import * as ImagePicker from "expo-image-picker";
import { summarizeSurgery, analyzeMedicalImage } from "../../../services/aiService";

const PRIMARY_COLOR = COLORS.primary;
const PRIMARY_LIGHT = "#E8F6F2";
const TAB_COLORS: Record<string, string> = {
  allergies: "#10B981",
  chronic: "#6366F1",
  medications: "#F59E0B",
  vitals: "#0EA5E9",
  surgeries: "#8B5CF6",
  documents: "#6366F1",
};

const TAB_BG: Record<string, string> = {
  allergies: "#ECFDF5",
  chronic: "#EEF2FF",
  medications: "#FFF7ED",
  vitals: "#F0F9FF",
  surgeries: "#F5F3FF",
  documents: "#EEF2FF",
};

export default function MedicalRecordsCategory() {
  const router = useRouter();
  const { category, folder } = useLocalSearchParams<{ category: string, folder?: string }>();
  const { tr, isRTL } = useLanguage();

  const [loading, setLoading] = useState(true);
  const [items, setItems] = useState<any[]>([]);

  // Add modal state
  const [showModal, setShowModal] = useState(false);
  const [saving, setSaving] = useState(false);

  // Allergy form
  const [allergyName, setAllergyName] = useState("");
  const [allergySeverity, setAllergySeverity] = useState("Mild");
  const [allergyNotes, setAllergyNotes] = useState("");

  // Surgery form
  const [surgeryName, setSurgeryName] = useState("");
  const [surgeryHospital, setSurgeryHospital] = useState("");
  const [surgeryDoctor, setSurgeryDoctor] = useState("");
  const [surgeryDate, setSurgeryDate] = useState("");
  const [surgeryComplications, setSurgeryComplications] = useState("");
  const [surgeryNotes, setSurgeryNotes] = useState("");

  // Chronic Disease form
  const [chronicName, setChronicName] = useState("");
  const [chronicType, setChronicType] = useState("");
  const [chronicSeverity, setChronicSeverity] = useState("Moderate");
  const [chronicFreq, setChronicFreq] = useState("");

  // Document form
  const [docTitle, setDocTitle] = useState("");
  const [docType, setDocType] = useState("Blood Test");
  const [docUri, setDocUri] = useState<string | null>(null);
  const [docDescription, setDocDescription] = useState("");
  const [selectedFolder, setSelectedFolder] = useState<string | null>(folder || null);
  const [isAiProcessing, setIsAiProcessing] = useState(false);

  const DOCUMENT_FOLDERS = [
    { id: "Blood Test", label: "Blood Tests", icon: "water-outline", color: "#6366F1" },
    { id: "X-Ray", label: "X-Rays / Scans", icon: "scan-outline", color: "#6366F1" },
    { id: "MRI", label: "MRI / CT", icon: "layers-outline", color: "#8B5CF6" },
    { id: "Prescription", label: "Prescriptions", icon: "receipt-outline", color: "#10B981" },
    { id: "Other", label: "Other", icon: "document-text-outline", color: "#64748B" },
  ];

  const canAdd = category !== "medications" && category !== "vitals";
  const color = TAB_COLORS[category as string] || COLORS.primary;

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const pid = await getMyPatientId();
      if (pid <= 0) { Toast.show({ type: "error", text1: tr("error") }); return; }
      let data: any[] = [];
      switch (category) {
        case "allergies": data = await getAllergies(pid); break;
        case "chronic": data = await getChronicDiseases(pid); break;
        case "medications": data = await getMedications(pid); break;
        case "vitals": data = await getVitals(pid); break;
        case "surgeries": data = await getSurgeries(pid); break;
        case "documents": data = await getPatientDocuments(pid); break;
      }
      setItems(data);
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || tr("failed_load_records") });
    } finally {
      setLoading(false);
    }
  }, [category]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleDelete = async (id: number) => {
    Alert.alert(tr("delete" as any), tr("delete_confirm" as any), [
      { text: tr("cancel"), style: "cancel" },
      {
        text: tr("delete" as any), style: "destructive", onPress: async () => {
          try {
            switch (category) {
              case "allergies": await deleteAllergy(id); break;
              case "chronic": await deleteChronicDisease(id); break;
              case "medications": await deleteMedication(id); break;
              case "vitals": await deleteVital(id); break;
              case "surgeries": await deleteSurgery(id); break;
              case "documents": await deletePatientDocument(id); break;
            }
            setItems(prev => prev.filter(x => x.id !== id));
            Toast.show({ type: "success", text1: tr("deleted" as any) });
          } catch (e: any) {
            Toast.show({ type: "error", text1: e.message || tr("error") });
          }
        }
      }
    ]);
  };

  const resetForms = () => {
    setAllergyName(""); setAllergySeverity("Mild"); setAllergyNotes("");
    setChronicName(""); setChronicType(""); setChronicSeverity("Moderate"); setChronicFreq("");
    setSurgeryName(""); setSurgeryHospital(""); setSurgeryDoctor(""); setSurgeryDate(""); setSurgeryComplications(""); setSurgeryNotes("");
    setDocTitle(""); setDocType("Blood Test"); setDocUri(null); setDocDescription("");
  };

  const handleAiRefineItem = async (type: string, currentVal: string, setter: (v: string) => void) => {
    if (!currentVal.trim()) {
      Alert.alert("Input Needed", "Please enter some text first.");
      return;
    }
    try {
      setIsAiProcessing(true);
      const { summarizeMedicalItem } = await import("../../../services/aiService");
      const refined = await summarizeMedicalItem(type, currentVal);
      const combinedText = `${refined.summary_en}\n---\n${refined.summary_ar}`;
      setter(combinedText);
      Toast.show({ type: "success", text1: "Refined by AI" });
    } catch (e) {
      Alert.alert("AI Error", "Could not refine text.");
    } finally {
      setIsAiProcessing(false);
    }
  };

  const handleAiAnalyzeDocument = async () => {
    if (!docUri) {
      Alert.alert("Input Needed", "Please select an image first.");
      return;
    }
    try {
      setIsAiProcessing(true);
      const type = docType === "Prescription" ? "prescription" : "lab";
      const result = await analyzeMedicalImage(docUri, type);
      
      // If it's a lab/prescription, we might want to pre-fill notes or title
      if (result.raw_text) {
        setDocTitle(`${result.summary_en} / ${result.summary_ar}`);
        setDocDescription(result.raw_text);
        Toast.show({ type: "success", text1: "AI Analysis Complete" });
        Alert.alert("AI Extraction", `${result.summary_en}\n---\n${result.summary_ar}`);
      }
    } catch (e) {
      Alert.alert("AI Error", "Could not analyze image at this time.");
    } finally {
      setIsAiProcessing(false);
    }
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      const pid = await getMyPatientId();
      if (pid <= 0) return;

      if (category === "allergies") {
        if (!allergyName.trim()) { Toast.show({ type: "error", text1: tr("please_enter_allergen" as any) }); return; }
        const res = await createAllergy(pid, { allergenName: allergyName, severity: allergySeverity, allergyType: "Unknown", isActive: true });
        setItems(prev => [res, ...prev]);
      } else if (category === "chronic") {
        if (!chronicName.trim()) { Toast.show({ type: "error", text1: tr("please_enter_disease" as any) }); return; }
        const res = await createChronicDisease(pid, { diseaseName: chronicName, diseaseType: chronicType, severity: chronicSeverity, monitoringFrequency: chronicFreq, isActive: true });
        setItems(prev => [res, ...prev]);
      } else if (category === "surgeries") {
        if (!surgeryName.trim()) { Toast.show({ type: "error", text1: tr("please_enter_surgery" as any) }); return; }
        const res = await createSurgery(pid, { surgeryName: surgeryName, hospitalName: surgeryHospital, doctorName: surgeryDoctor, surgeryDate: surgeryDate, complications: surgeryComplications, notes: surgeryNotes });
        setItems(prev => [res, ...prev]);
      } else if (category === "documents") {
        if (!docTitle.trim()) { Toast.show({ type: "error", text1: tr("please_enter_title" as any) }); return; }
        if (!docUri) { Toast.show({ type: "error", text1: tr("please_select_image" as any) }); return; }
        const res = await uploadPatientDocument(pid, docUri, docType.toLowerCase(), docTitle, docDescription);
        setItems(prev => [res, ...prev]);
      }

      Toast.show({ type: "success", text1: tr("saved" as any) });
      setShowModal(false);
      resetForms();
    } catch (e: any) {
      console.log("Upload error details:", e);
      Toast.show({ type: "error", text1: e.message || tr("error") });
    } finally {
      setSaving(false);
    }
  };

  const pickImage = async () => {
    Alert.alert(
      tr("select_image" as any),
      "Would you like to take a new photo or choose from your gallery?",
      [
        { text: "Take Photo", onPress: () => handleImageSource(true) },
        { text: "Choose from Gallery", onPress: () => handleImageSource(false) },
        { text: "Cancel", style: "cancel" }
      ]
    );
  };

  const handleImageSource = async (useCamera: boolean) => {
    try {
      if (useCamera) {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== 'granted') {
          Alert.alert("Permission Needed", "Camera access is required to take photos.");
          return;
        }
      }

      const options: ImagePicker.ImagePickerOptions = {
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: false, // Don't crop, keep full image
        quality: 0.9, // Higher quality for better AI readability
      };

      const result = useCamera
        ? await ImagePicker.launchCameraAsync(options)
        : await ImagePicker.launchImageLibraryAsync(options);

      if (!result.canceled && result.assets && result.assets[0].uri) {
        setDocUri(result.assets[0].uri);
      }
    } catch (e: any) {
      Toast.show({ type: "error", text1: e.message || "Failed to pick image" });
    }
  };

  const renderItem = (item: any, index: number) => {
    const itemColor = color;
    switch (category) {
      case "allergies":
        return (
          <View key={item.id || index} style={[styles.card, { borderLeftColor: itemColor }]}>
            <View style={styles.cardRow}>
              <Text style={styles.cardTitle}>{item.allergenName}</Text>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
                <View style={[styles.badge, { backgroundColor: itemColor + '20' }]}>
                  <Text style={[styles.badgeTxt, { color: itemColor }]}>{item.severity}</Text>
                </View>
                <TouchableOpacity onPress={() => handleDelete(item.id)}>
                  <Ionicons name="trash-outline" size={18} color="#EF4444" />
                </TouchableOpacity>
              </View>
            </View>
            {item.notes && <Text style={styles.cardMeta}>{item.notes}</Text>}
          </View>
        );
      case "chronic":
        return (
          <View key={item.id || index} style={[styles.card, { borderLeftColor: itemColor }]}>
            <View style={styles.cardRow}>
              <Text style={styles.cardTitle}>{item.diseaseName}</Text>
              <View style={[styles.badge, { backgroundColor: item.isActive ? itemColor + '20' : "#F1F5F9" }]}>
                <Text style={[styles.badgeTxt, { color: item.isActive ? itemColor : "#64748B" }]}>{item.isActive ? "Active" : "Inactive"}</Text>
              </View>
            </View>
            {item.diagnosedDate && <Text style={styles.cardMeta}>{tr("date")}: {item.diagnosedDate}</Text>}
          </View>
        );
      case "medications":
        return (
          <View key={item.id || index} style={[styles.card, { borderLeftColor: itemColor }]}>
            <View style={styles.cardRow}>
              <Text style={styles.cardTitle}>{item.medicationName}</Text>
              <View style={[styles.badge, { backgroundColor: item.isActive ? itemColor + '20' : "#F1F5F9" }]}>
                <Text style={[styles.badgeTxt, { color: item.isActive ? itemColor : "#64748B" }]}>{item.isActive ? "Active" : "Inactive"}</Text>
              </View>
            </View>
            <Text style={styles.cardText}>{item.dosage} • {item.frequency}</Text>
            {item.instructions && <Text style={styles.cardMeta}>{item.instructions}</Text>}
          </View>
        );
      case "vitals":
        return (
          <View key={item.id || index} style={[styles.card, { borderLeftColor: itemColor }]}>
            <View style={styles.cardRow}>
              <Text style={styles.cardTitle}>{item.readingType}</Text>
              <Text style={[styles.vitalValue, !item.isNormal && { color: "#C2410C" }]}>
                {item.value}{item.value2 ? `/${item.value2}` : ""} {item.unit}
              </Text>
            </View>
            {item.notes && <Text style={styles.cardMeta}>{item.notes}</Text>}
            {item.recordedAt && <Text style={styles.cardMeta}>{new Date(item.recordedAt).toLocaleDateString()}</Text>}
          </View>
        );
      case "surgeries":
        return (
          <View key={item.id || index} style={[styles.card, { borderLeftColor: itemColor }]}>
            <View style={styles.cardRow}>
              <Text style={styles.cardTitle}>{item.surgeryName}</Text>
              <TouchableOpacity onPress={() => handleDelete(item.id)}>
                <Ionicons name="trash-outline" size={18} color="#EF4444" />
              </TouchableOpacity>
            </View>
            {item.hospitalName && <Text style={styles.cardText}>{tr("hospital" as any)}: {item.hospitalName}</Text>}
            {item.doctorName && <Text style={styles.cardText}>{tr("doctor")}: {item.doctorName}</Text>}
            {item.surgeryDate && <Text style={styles.cardMeta}>{tr("date")}: {item.surgeryDate}</Text>}
          </View>
        );
      case "documents":
        return (
          <View key={item.id || index} style={[styles.card, { borderLeftColor: itemColor }]}>
            <View style={styles.cardRow}>
              <Text style={styles.cardTitle}>{item.title}</Text>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
                <View style={[styles.badge, { backgroundColor: itemColor + '20' }]}>
                  <Text style={[styles.badgeTxt, { color: itemColor }]}>{item.documentType}</Text>
                </View>
                <TouchableOpacity onPress={() => handleDelete(item.id)}>
                  <Ionicons name="trash-outline" size={18} color="#EF4444" />
                </TouchableOpacity>
              </View>
            </View>
            {item.fileUrl && (
              <TouchableOpacity onPress={() => { /* View doc */ }}>
                <Text style={[styles.linkTxt, { color: itemColor }]}>View Document</Text>
              </TouchableOpacity>
            )}
            {item.description && (
              <View style={styles.aiDescBox}>
                <Sparkles size={14} color="#7C3AED" />
                <Text style={styles.aiDescTxt}>{item.description}</Text>
              </View>
            )}
          </View>
        );
      default: return null;
    }
  };

  const renderModal = () => {
    if (!showModal) return null;
    return (
      <Modal visible={showModal} animationType="slide" transparent onRequestClose={() => setShowModal(false)}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>
              {category === "allergies" ? tr("add_allergy" as any) : 
               category === "chronic" ? tr("add_chronic" as any) :
               category === "surgeries" ? tr("add_surgery" as any) : tr("add_document" as any)}
            </Text>
            <ScrollView showsVerticalScrollIndicator={false}>
              {category === "allergies" && (
                <>
                  <View style={styles.inputWithAi}>
                    <TextInput style={[styles.input, { flex: 1, marginBottom: 0 }]} placeholder={tr("allergen_name" as any)} value={allergyName} onChangeText={setAllergyName} />
                    <TouchableOpacity style={[styles.aiButtonSmall, { backgroundColor: TAB_COLORS.allergies }]} onPress={() => handleAiRefineItem("allergy", allergyName, setAllergyName)}>
                      <Ionicons name="sparkles" size={18} color="#fff" />
                    </TouchableOpacity>
                  </View>
                  <TextInput style={styles.input} placeholder={tr("severity" as any)} value={allergySeverity} onChangeText={setAllergySeverity} />
                  <TextInput style={styles.input} placeholder={tr("notes" as any)} value={allergyNotes} onChangeText={setAllergyNotes} multiline />
                </>
              )}
              {category === "chronic" && (
                <>
                  <TextInput style={styles.input} placeholder={tr("disease_name" as any)} value={chronicName} onChangeText={setChronicName} />
                  <TextInput style={styles.input} placeholder={tr("disease_type" as any)} value={chronicType} onChangeText={setChronicType} />
                  <TextInput style={styles.input} placeholder={tr("severity" as any)} value={chronicSeverity} onChangeText={setChronicSeverity} />
                  <TextInput style={styles.input} placeholder={tr("monitoring_frequency" as any)} value={chronicFreq} onChangeText={setChronicFreq} />
                </>
              )}
              {category === "surgeries" && (
                <>
                  <TextInput style={styles.input} placeholder={tr("surgery_name" as any)} value={surgeryName} onChangeText={setSurgeryName} />
                  <TextInput style={styles.input} placeholder={tr("hospital" as any)} value={surgeryHospital} onChangeText={setSurgeryHospital} />
                  <TextInput style={styles.input} placeholder={tr("doctor_name" as any)} value={surgeryDoctor} onChangeText={setSurgeryDoctor} />
                  <TextInput style={styles.input} placeholder={tr("date" as any)} value={surgeryDate} onChangeText={setSurgeryDate} />
                  <TextInput style={styles.input} placeholder={tr("complications" as any)} value={surgeryComplications} onChangeText={setSurgeryComplications} multiline />
                  <View style={styles.inputWithAi}>
                    <TextInput style={[styles.input, { flex: 1, marginBottom: 0 }]} placeholder={tr("notes" as any)} value={surgeryNotes} onChangeText={setSurgeryNotes} multiline />
                    <TouchableOpacity style={styles.aiButtonSmall} onPress={() => handleAiRefineItem("surgery description", surgeryNotes, setSurgeryNotes)} disabled={isAiProcessing}>
                      {isAiProcessing ? <ActivityIndicator size="small" color="#fff" /> : <Ionicons name="sparkles" size={18} color="#fff" />}
                    </TouchableOpacity>
                  </View>
                  <Text style={styles.aiHint}>Use AI to refine and summarize notes</Text>
                </>
              )}
              {category === "documents" && (
                <>
                  <View style={styles.inputWithAi}>
                    <TextInput style={[styles.input, { flex: 1, marginBottom: 0 }]} placeholder={tr("title" as any)} value={docTitle} onChangeText={setDocTitle} />
                    {docUri && (
                      <TouchableOpacity style={[styles.aiButtonSmall, { backgroundColor: color }]} onPress={handleAiAnalyzeDocument} disabled={isAiProcessing}>
                        {isAiProcessing ? <ActivityIndicator size="small" color="#fff" /> : <Ionicons name="sparkles" size={18} color="#fff" />}
                      </TouchableOpacity>
                    )}
                  </View>
                  {docUri && <Text style={styles.aiHint}>AI can analyze image to extract title & description</Text>}
                  <Text style={styles.inputLabelSmall}>Select Folder</Text>
                  <View style={styles.folderPickerRow}>
                    {DOCUMENT_FOLDERS.map(f => (
                      <TouchableOpacity 
                        key={f.id} 
                        style={[styles.folderOption, docType === f.id && { borderColor: f.color, backgroundColor: f.color + '10' }]}
                        onPress={() => setDocType(f.id)}
                      >
                        <Ionicons name={f.icon as any} size={20} color={docType === f.id ? f.color : "#94A3B8"} />
                        <Text style={[styles.folderOptionTxt, docType === f.id && { color: f.color }]}>{f.label}</Text>
                      </TouchableOpacity>
                    ))}
                  </View>
                  <TouchableOpacity style={styles.imageBtn} onPress={pickImage}>
                    <Ionicons name="camera-outline" size={24} color={color} />
                    <Text style={[styles.imageBtnTxt, { color }]}>{docUri ? tr("change_image" as any) : tr("select_image" as any)}</Text>
                  </TouchableOpacity>
                  {docUri && (
                    <TextInput 
                      style={[styles.input, { height: 100, textAlignVertical: 'top' }]} 
                      placeholder="Document Description / AI Analysis" 
                      value={docDescription} 
                      onChangeText={setDocDescription} 
                      multiline 
                    />
                  )}
                  {docUri && <Text style={styles.imageUri} numberOfLines={1}>{docUri}</Text>}
                </>
              )}
            </ScrollView>
            <View style={styles.modalActions}>
              <TouchableOpacity style={[styles.btn, styles.btnSecondary]} onPress={() => { setShowModal(false); resetForms(); }}>
                <Text style={styles.btnSecondaryTxt}>{tr("cancel")}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.btn, styles.btnPrimary, { backgroundColor: color }]} onPress={handleSave} disabled={saving}>
                {saving ? <ActivityIndicator color="#fff" size="small" /> : <Text style={styles.btnPrimaryTxt}>{tr("save")}</Text>}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    );
  };

  if (loading) {
    return (
      <View style={[styles.container, { justifyContent: "center", alignItems: "center" }]}>
        <StatusBar barStyle="light-content" />
        <ActivityIndicator size="large" color={color} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />
      <LinearGradient colors={[color, color + 'CC']} style={styles.header}>
        <View style={styles.headerTop}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
            <Ionicons name="arrow-back" size={24} color="#fff" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{tr(category as any)}</Text>
          <View style={{ width: 44 }} />
        </View>
      </LinearGradient>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        {category === "documents" && !selectedFolder ? (
          <View style={styles.foldersGrid}>
            {DOCUMENT_FOLDERS.map(f => (
              <TouchableOpacity 
                key={f.id} 
                style={styles.folderCard} 
                onPress={() => setSelectedFolder(f.id)}
              >
                <View style={[styles.folderIconBox, { backgroundColor: f.color + '15' }]}>
                  <Ionicons name={f.icon as any} size={32} color={f.color} />
                </View>
                <Text style={styles.folderName}>{f.label}</Text>
                <Text style={styles.folderCount}>{items.filter(it => it.documentType === f.id).length} Files</Text>
              </TouchableOpacity>
            ))}
          </View>
        ) : (
          <>
            {category === "documents" && selectedFolder && (
              <TouchableOpacity style={styles.backToFolders} onPress={() => setSelectedFolder(null)}>
                <Ionicons name="chevron-back" size={18} color={color} />
                <Text style={[styles.backToFoldersTxt, { color }]}>Back to Folders</Text>
              </TouchableOpacity>
            )}
            {items.length === 0 || (category === "documents" && items.filter(it => it.documentType === selectedFolder).length === 0) ? (
              <View style={styles.emptyState}>
                <View style={[styles.emptyCircle, { backgroundColor: TAB_BG[category as string] || "#F1F5F9" }]}>
                  <Ionicons name={
                    category === "allergies" ? "warning-outline" :
                    category === "chronic" ? "fitness-outline" :
                    category === "medications" ? "medical-outline" :
                    category === "vitals" ? "pulse-outline" :
                    category === "surgeries" ? "cut-outline" : "document-text-outline"
                  } size={32} color={color} />
                </View>
                <Text style={styles.emptyStateTitle}>{tr("no_records" as any)}</Text>
                <Text style={styles.emptyStateSub}>{tr("no_records_sub" as any)}</Text>
              </View>
            ) : (
              (category === "documents" ? items.filter(it => it.documentType === selectedFolder) : items)
                .map((item, i) => renderItem(item, i))
            )}
          </>
        )}
      </ScrollView>
      {canAdd && (
        <TouchableOpacity 
          style={[styles.fab, { backgroundColor: color }]} 
          onPress={() => setShowModal(true)}
          activeOpacity={0.8}
        >
          <LinearGradient colors={[color, color + 'CC']} style={styles.fabGradient}>
            <Ionicons name="add" size={32} color="#fff" />
          </LinearGradient>
        </TouchableOpacity>
      )}
      {renderModal()}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  header: { paddingTop: 60, paddingBottom: 25, borderBottomLeftRadius: 35, borderBottomRightRadius: 35, elevation: 10, shadowOpacity: 0.1 },
  headerTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingHorizontal: 20 },
  backBtn: { width: 44, height: 44, borderRadius: 15, backgroundColor: 'rgba(255,255,255,0.2)', justifyContent: 'center', alignItems: 'center' },
  headerTitle: { fontSize: 20, fontWeight: "800", color: "#fff", letterSpacing: 0.5 },
  scroll: { flex: 1 },
  scrollContent: { padding: 20, paddingBottom: 100 },
  card: { backgroundColor: "#fff", borderRadius: 24, padding: 20, marginBottom: 16, shadowColor: "#000", shadowOpacity: 0.04, shadowRadius: 15, elevation: 5, borderLeftWidth: 6 },
  cardRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  cardTitle: { fontSize: 17, fontWeight: "800", color: "#1E293B", flex: 1 },
  cardText: { fontSize: 14, color: "#475569", marginTop: 5, fontWeight: '500', lineHeight: 20 },
  cardMeta: { fontSize: 12, color: "#94A3B8", marginTop: 8, fontWeight: '600' },
  badge: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12 },
  badgeTxt: { fontSize: 10, fontWeight: "800" },
  vitalValue: { fontSize: 20, fontWeight: "900", color: COLORS.primary },
  linkTxt: { fontSize: 13, fontWeight: "700", marginTop: 10 },
  emptyState: { alignItems: "center", paddingVertical: 60 },
  emptyCircle: { width: 90, height: 90, borderRadius: 32, justifyContent: "center", alignItems: "center", marginBottom: 20, elevation: 4 },
  emptyStateTitle: { fontSize: 18, fontWeight: "800", color: "#1E293B", marginBottom: 8 },
  emptyStateSub: { fontSize: 14, color: "#94A3B8", textAlign: "center", maxWidth: 280, lineHeight: 22 },
  fab: { position: "absolute", right: 25, bottom: 40, width: 64, height: 64, borderRadius: 32, elevation: 12, shadowOpacity: 0.3, shadowRadius: 15, overflow: 'hidden' },
  fabGradient: { width: '100%', height: '100%', justifyContent: 'center', alignItems: 'center' },
  modalOverlay: { flex: 1, backgroundColor: "rgba(6, 78, 59, 0.4)", justifyContent: "flex-end" },
  modalContent: { backgroundColor: "#fff", borderTopLeftRadius: 40, borderTopRightRadius: 40, padding: 25, paddingBottom: 45, elevation: 25 },
  modalTitle: { fontSize: 19, fontWeight: "800", color: "#1E293B", marginBottom: 25, textAlign: 'center' },
  input: { backgroundColor: "#F8FAFC", borderRadius: 18, paddingHorizontal: 18, paddingVertical: 15, marginBottom: 15, fontSize: 15, color: "#1E293B", borderWidth: 1, borderColor: "#F1F5F9", fontWeight: '600' },
  imageBtn: { flexDirection: "row", alignItems: "center", gap: 10, paddingVertical: 15, borderRadius: 18, borderWidth: 1, borderStyle: "dashed", justifyContent: "center", marginBottom: 15, backgroundColor: '#F8FAFC' },
  imageBtnTxt: { fontSize: 15, fontWeight: "700" },
  imageUri: { fontSize: 12, color: "#64748B", marginBottom: 15, textAlign: "center", fontWeight: '500' },
  modalActions: { flexDirection: "row", justifyContent: "center", gap: 15, marginTop: 15 },
  btn: { flex: 1, height: 56, borderRadius: 18, justifyContent: 'center', alignItems: 'center' },
  btnPrimary: { elevation: 8, shadowOpacity: 0.3 },
  btnPrimaryTxt: { color: "#fff", fontWeight: "800", fontSize: 15 },
  btnSecondary: { backgroundColor: "#F1F5F9" },
  btnSecondaryTxt: { color: "#475569", fontWeight: "800", fontSize: 15 },

  // Folder Styles
  foldersGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', gap: 15 },
  folderCard: { width: '47%', backgroundColor: '#fff', borderRadius: 24, padding: 20, alignItems: 'center', elevation: 4, shadowOpacity: 0.05, borderWidth: 1, borderColor: '#F1F5F9' },
  folderIconBox: { width: 64, height: 64, borderRadius: 20, justifyContent: 'center', alignItems: 'center', marginBottom: 12 },
  folderName: { fontSize: 14, fontWeight: '800', color: '#1E293B', textAlign: 'center' },
  folderCount: { fontSize: 11, color: '#94A3B8', marginTop: 4, fontWeight: '600' },
  backToFolders: { flexDirection: 'row', alignItems: 'center', gap: 5, marginBottom: 15, paddingLeft: 5 },
  backToFoldersTxt: { fontSize: 14, fontWeight: '700' },
  folderPickerRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginBottom: 20, marginTop: 5 },
  folderOption: { paddingHorizontal: 12, paddingVertical: 10, borderRadius: 12, borderWidth: 1, borderColor: '#F1F5F9', backgroundColor: '#F8FAFC', flexDirection: 'row', alignItems: 'center', gap: 8 },
  folderOptionTxt: { fontSize: 11, fontWeight: '700', color: '#64748B' },
  inputLabelSmall: { fontSize: 13, fontWeight: '800', color: '#64748B', marginBottom: 8, marginLeft: 5 },
  inputWithAi: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 5 },
  aiButtonSmall: { width: 44, height: 44, borderRadius: 12, backgroundColor: '#8B5CF6', justifyContent: 'center', alignItems: 'center', elevation: 4 },
  aiHint: { fontSize: 11, color: '#8B5CF6', fontWeight: '700', marginLeft: 5, marginBottom: 15 },
  aiButtonLarge: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 10, paddingVertical: 12, borderRadius: 18, marginBottom: 15, elevation: 6 },
  aiButtonLargeTxt: { color: '#fff', fontSize: 14, fontWeight: '800' },
  aiDescBox: { marginTop: 12, padding: 12, backgroundColor: '#F5F3FF', borderRadius: 16, flexDirection: 'row', gap: 10, borderWidth: 1, borderColor: '#EDE9FE' },
  aiDescTxt: { fontSize: 11, color: '#5B21B6', flex: 1, lineHeight: 16, fontWeight: '600' },
});
