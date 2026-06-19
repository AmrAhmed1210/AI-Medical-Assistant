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
        DateTime? ClosedAt,
        bool? FollowUpRequired,
        string? FollowUpDate,
        string? FollowUpTime,
        string? FollowUpNotes = null,
        string? DoctorName = null,
        string? DoctorSpecialty = null,
        bool IsFollowUp = false,
        int? FollowUpVisitId = null
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
        string? Notes,
        List<UpdateSymptomDto>? Symptoms,
        List<UpdatePrescriptionDto>? Prescriptions,
        List<UpdateVitalDto>? VitalSigns,
        bool? FollowUpRequired,
        string? FollowUpDate,
        string? FollowUpTime,
        string? FollowUpNotes
    );

    public record UpdateSymptomDto(
        string Name,
        string Severity,
        string Onset,
        string? Location,
        string? Duration,
        bool IsChronic
    );

    public record UpdatePrescriptionDto(
        string MedicationName,
        string Dosage,
        string Frequency,
        string Duration,
        int? Quantity,
        string? Instructions,
        bool IsChronic
    );

    public record UpdateVitalDto(
        string Type,
        decimal Value,
        decimal? Value2,
        string Unit,
        bool IsAbnormal
    );

    public record VisitSummaryDto(
        int Id,
        string PatientName,
        int PatientAge,
        string BloodType,
        List<AllergySummaryDto> Allergies,
        DateOnly VisitDate,
        string ChiefComplaint,
        string? ExaminationFindings,
        string? Assessment,
        string? Plan,
        List<VitalSummaryDto> VitalSigns,
        List<PrescriptionSummaryDto> Prescriptions,
        List<SymptomSummaryDto> Symptoms,
        string? Notes,
        bool? FollowUpRequired,
        string? FollowUpDate,
        string? FollowUpTime,
        string? FollowUpNotes,
        List<LastVisitSummaryDto> RecentVisits,
        string? VisitsTimelineSummaryEn,
        string? VisitsTimelineSummaryAr
    );

    public record AllergySummaryDto(
        string AllergenName,
        string Severity,
        string Reaction
    );

    public record VitalSummaryDto(
        string Type,
        decimal Value,
        decimal? Value2,
        string Unit,
        bool IsAbnormal
    );

    public record PrescriptionSummaryDto(
        string MedicationName,
        string Dosage,
        string Frequency,
        string Duration,
        int? Quantity,
        string? Instructions,
        bool IsChronic
    );

    public record PatientHistoryDto(
        string BloodType,
        List<AllergySummaryDto> Allergies,
        List<ChronicDiseaseSummaryDto> ChronicDiseases,
        List<MedicationSummaryDto> Medications,
        Dictionary<string, string> LatestVitals,
        List<LastVisitSummaryDto> LastVisits,
        string? AiDiagnosisSummary
    );

    public record ChronicDiseaseSummaryDto(
        string Id,
        string DiseaseName,
        string? TargetValues
    );

    public record MedicationSummaryDto(
        string Id,
        string MedicationName,
        string Dosage,
        string Form
    );

    public record LastVisitSummaryDto(
        string Id,
        string VisitDate,
        string ChiefComplaint,
        string? DoctorName = null,
        string? DoctorSpecialty = null,
        string? Summary = null,
        string? SummaryEn = null,
        string? SummaryAr = null
    );

    public record SymptomSummaryDto(
        string Name,
        string Severity,
        string Onset,
        string? Location,
        string? Duration,
        bool IsChronic
    );
}
