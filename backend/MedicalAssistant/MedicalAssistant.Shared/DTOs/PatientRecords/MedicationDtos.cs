using System;

namespace MedicalAssistant.Shared.DTOs.PatientRecords
{
    public record MedicationTrackerDto(
        int Id,
        int PatientId,
        int? PrescribedByDoctorId,
        int? ChronicDiseaseMonitorId,
        string MedicationName,
        string? GenericName,
        string Dosage,
        string Form,
        string Frequency,
        int TimesPerDay,
        string DoseTimes,
        DateOnly StartDate,
        DateOnly? EndDate,
        string? Instructions,
        int? PillsRemaining,
        int RefillThreshold,
        bool IsChronic,
        bool IsActive,
        DateTime CreatedAt
    );

    public record CreateMedicationTrackerDto(
        int? ChronicDiseaseMonitorId,
        string MedicationName,
        string? GenericName,
        string Dosage,
        string Form,
        string Frequency,
        int TimesPerDay,
        string DoseTimes,
        DateOnly StartDate,
        DateOnly? EndDate,
        string? Instructions,
        int? PillsRemaining,
        int RefillThreshold,
        bool IsChronic,
        bool? IsActive
    );

    public record UpdateMedicationTrackerDto(
        int? ChronicDiseaseMonitorId,
        string? MedicationName,
        string? GenericName,
        string? Dosage,
        string? Form,
        string? Frequency,
        int? TimesPerDay,
        string? DoseTimes,
        DateOnly? StartDate,
        DateOnly? EndDate,
        string? Instructions,
        int? PillsRemaining,
        int? RefillThreshold,
        bool? IsChronic,
        bool? IsActive
    );

    public record MedicationScheduleItemDto(
        int MedicationTrackerId,
        DateTime ScheduledAt,
        string MedicationName,
        string Dosage,
        string Status
    );
}
