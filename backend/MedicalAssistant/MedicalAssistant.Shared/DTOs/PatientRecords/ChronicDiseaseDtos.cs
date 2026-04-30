using System;

namespace MedicalAssistant.Shared.DTOs.PatientRecords
{
    public record ChronicDiseaseMonitorDto(
        int Id,
        int PatientId,
        string DiseaseName,
        string DiseaseType,
        DateOnly? DiagnosedDate,
        string Severity,
        bool IsActive,
        string? DoctorNotes,
        string? TargetValues,
        string MonitoringFrequency,
        DateOnly? LastCheckDate,
        DateTime CreatedAt,
        DateTime UpdatedAt
    );

    public record CreateChronicDiseaseMonitorDto(
        string DiseaseName,
        string DiseaseType,
        DateOnly? DiagnosedDate,
        string Severity,
        string MonitoringFrequency,
        string? DoctorNotes,
        string? TargetValues,
        DateOnly? LastCheckDate,
        bool? IsActive
    );

    public record UpdateChronicDiseaseMonitorDto(
        string? DiseaseName,
        string? DiseaseType,
        DateOnly? DiagnosedDate,
        string? Severity,
        string? MonitoringFrequency,
        string? DoctorNotes,
        string? TargetValues,
        DateOnly? LastCheckDate,
        bool? IsActive
    );
}
