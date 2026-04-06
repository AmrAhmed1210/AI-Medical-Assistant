using System;
using System.Collections.Generic;

namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class AIReportDto
{
    // التعديل: تغيير النوع من Guid إلى int ليتوافق مع قاعدة البيانات
    public int Id { get; set; }

    public int PatientId { get; set; }

    public string PatientName { get; set; } = string.Empty;

    public string UrgencyLevel { get; set; } = string.Empty; // HIGH, MEDIUM, LOW

    public List<SymptomDto> Symptoms { get; set; } = new();

    public string Disclaimer { get; set; } = "This is not medical advice.";

    public DateTime CreatedAt { get; set; }
}