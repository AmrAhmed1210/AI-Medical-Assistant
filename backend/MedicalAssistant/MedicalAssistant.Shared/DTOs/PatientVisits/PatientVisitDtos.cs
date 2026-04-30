using System;

namespace MedicalAssistant.Shared.DTOs.PatientVisits
{
    public record PatientVisitDto(
        int Id,
        int PatientId,
        int DoctorId,
        int? AppointmentId,
        string ChiefComplaint,
        string? PresentIllnessHistory,
        string? ExaminationFindings,
        string? Assessment,
        string? Plan,
        string? Notes,
        string Status,
        DateOnly VisitDate,
        DateTime CreatedAt,
        DateTime? ClosedAt
    );

    public record CreateVisitDto(
        int PatientId,
        int? AppointmentId,
        string ChiefComplaint,
        string? PresentIllnessHistory
    );

    public record UpdateVisitDto(
        string? ChiefComplaint,
        string? PresentIllnessHistory,
        string? ExaminationFindings,
        string? Assessment,
        string? Plan,
        string? Notes
    );

    public record VisitSummaryDto(
        int VisitId,
        int PatientId,
        int DoctorId,
        DateOnly VisitDate,
        string ChiefComplaint,
        string? SummarySnapshot
    );
}
