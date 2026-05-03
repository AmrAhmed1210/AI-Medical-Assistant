using System;

namespace MedicalAssistant.Shared.DTOs.PatientVisits
{
    public record VisitPrescriptionDto(
        int Id,
        int PatientVisitId,
        string MedicationName,
        string? GenericName,
        string Dosage,
        string Form,
        string Frequency,
        int TimesPerDay,
        string? SpecificTimes,
        string? Duration,
        int? Quantity,
        string? Instructions,
        bool IsChronic,
        int Refills,
        bool IsDispensed,
        string? Notes,
        DateTime CreatedAt
    );

    public record CreateVisitPrescriptionDto(
        string MedicationName,
        string? GenericName,
        string Dosage,
        string Form,
        string Frequency,
        int TimesPerDay,
        string? SpecificTimes,
        string? Duration,
        int? Quantity,
        string? Instructions,
        bool IsChronic,
        int Refills,
        string? Notes
    );

    public record UpdateVisitPrescriptionDto(
        string? MedicationName,
        string? GenericName,
        string? Dosage,
        string? Form,
        string? Frequency,
        int? TimesPerDay,
        string? SpecificTimes,
        string? Duration,
        int? Quantity,
        string? Instructions,
        bool? IsChronic,
        int? Refills,
        string? Notes
    );
}
