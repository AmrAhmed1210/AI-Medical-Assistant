using System;
using System.Collections.Generic;

namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class AIReportDto
{
    public int Id { get; set; }

    /// <summary>String alias used by the frontend (maps to Id)</summary>
    public string ReportId => Id.ToString();

    public int PatientId { get; set; }

    public string PatientName { get; set; } = string.Empty;

    /// <summary>Session that generated this report</summary>
    public int SessionId { get; set; }

    public string UrgencyLevel { get; set; } = string.Empty; // HIGH, MEDIUM, LOW

    public List<SymptomDto> Symptoms { get; set; } = new();

    public string Disclaimer { get; set; } = "This is not medical advice.";

    /// <summary>Arabic disclaimer – generated from Disclaimer when null</summary>
    public string? DisclaimerAr { get; set; }

    public DateTime CreatedAt { get; set; }
}
