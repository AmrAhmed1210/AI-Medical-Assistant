using System;

namespace MedicalAssistant.Shared.DTOs.PatientVisits
{
    public record VisitSymptomDto(
        int Id,
        int PatientVisitId,
        string Name,
        string Severity,
        string? Duration,
        string? Onset,
        string? Progression,
        string? Location,
        bool IsChronic,
        string? Notes,
        DateTime CreatedAt
    );

    public record CreateVisitSymptomDto(
        string Name,
        string Severity,
        string? Duration,
        string? Onset,
        string? Progression,
        string? Location,
        bool IsChronic,
        string? Notes
    );
}
