import json
import re

file_path = "e:/AI/DoctorApp/app/(patient)/medical-records/[category].tsx"
i18n_path = "e:/AI/DoctorApp/constants/i18n.ts"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Strings to replace
replacements = {
    '"Please enter value"': 'tr(isRTL ? "ar" : "en", "please_enter_value" as any)',
    '"Please enter name"': 'tr(isRTL ? "ar" : "en", "please_enter_name" as any)',
    '"Record updated"': 'tr(isRTL ? "ar" : "en", "record_updated" as any)',
    '"Record added"': 'tr(isRTL ? "ar" : "en", "record_added" as any)',
    '"Could not save record."': 'tr(isRTL ? "ar" : "en", "could_not_save_record" as any)',
    '"Take Photo"': 'tr(isRTL ? "ar" : "en", "take_photo" as any)',
    '"Choose from Gallery"': 'tr(isRTL ? "ar" : "en", "choose_from_gallery" as any)',
    '"Cancel"': 'tr(isRTL ? "ar" : "en", "cancel" as any)',
    '"Error"': 'tr(isRTL ? "ar" : "en", "error" as any)',
    '"Would you like to take a new photo or choose from your gallery?"': 'tr(isRTL ? "ar" : "en", "photo_prompt" as any)',
    '"Permission Needed"': 'tr(isRTL ? "ar" : "en", "permission_needed" as any)',
    '"Camera access is required to take photos."': 'tr(isRTL ? "ar" : "en", "camera_access_required" as any)',
    '"Failed to pick image"': 'tr(isRTL ? "ar" : "en", "failed_pick_image" as any)'
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

# Now append to i18n
import os
new_en = """
  please_enter_value: "Please enter value",
  please_enter_name: "Please enter name",
  record_updated: "Record updated successfully",
  record_added: "Record added successfully",
  could_not_save_record: "Could not save record",
  take_photo: "Take Photo",
  choose_from_gallery: "Choose from Gallery",
  cancel: "Cancel",
  error: "Error",
  photo_prompt: "Would you like to take a new photo or choose from your gallery?",
  permission_needed: "Permission Needed",
  camera_access_required: "Camera access is required to take photos.",
  failed_pick_image: "Failed to pick image",
"""

new_ar = """
  please_enter_value: "يرجى إدخال القيمة",
  please_enter_name: "يرجى إدخال الاسم",
  record_updated: "تم تحديث السجل بنجاح",
  record_added: "تم إضافة السجل بنجاح",
  could_not_save_record: "تعذر حفظ السجل",
  take_photo: "التقاط صورة",
  choose_from_gallery: "اختيار من المعرض",
  cancel: "إلغاء",
  error: "خطأ",
  photo_prompt: "هل تود التقاط صورة جديدة أم الاختيار من المعرض؟",
  permission_needed: "مطلوب إذن",
  camera_access_required: "مطلوب إذن الوصول للكاميرا لالتقاط الصور.",
  failed_pick_image: "فشل في التقاط الصورة",
"""

with open(i18n_path, "r", encoding="utf-8") as f:
    i18n_content = f.read()

# Insert before `}` in en section and ar section
i18n_content = i18n_content.replace('profile: "Profile",', f'profile: "Profile",{new_en}')
i18n_content = i18n_content.replace('profile: "الملف الشخصي",', f'profile: "الملف الشخصي",{new_ar}')

with open(i18n_path, "w", encoding="utf-8") as f:
    f.write(i18n_content)

print("Strings replaced and i18n updated successfully.")
