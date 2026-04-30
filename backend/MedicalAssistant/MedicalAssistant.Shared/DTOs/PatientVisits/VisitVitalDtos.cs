using System;

namespace MedicalAssistant.Shared.DTOs.PatientVisits
{
    public record VisitVitalDto(
        int Id,
        int PatientId,
        int PatientVisitId,
        string Type,
        decimal Value,
        decimal? Value2,
        string Unit,
        bool IsAbnormal,
        decimal? NormalRangeMin,
        decimal? NormalRangeMax,
        string RecordedBy,
        string? Notes,
        DateTime RecordedAt
    );

    public record CreateVisitVitalDto(
        string Type,
        decimal Value,
        decimal? Value2,
        string Unit,
        bool IsAbnormal,
        decimal? NormalRangeMin,
        decimal? NormalRangeMax,
        string? Notes
    );
}
