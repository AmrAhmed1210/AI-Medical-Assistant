using System;

namespace MedicalAssistant.Shared.DTOs.PatientRecords
{
    public record VitalReadingDto(
        int Id,
        int PatientId,
        int? ChronicDiseaseMonitorId,
        string ReadingType,
        decimal Value,
        decimal? Value2,
        string Unit,
        string? SugarReadingContext,
        bool IsNormal,
        string RecordedBy,
        string? Notes,
        DateTime RecordedAt
    );

    public record CreateVitalReadingDto(
        int? ChronicDiseaseMonitorId,
        string ReadingType,
        decimal Value,
        decimal? Value2,
        string Unit,
        string? SugarReadingContext,
        bool IsNormal,
        string? Notes,
        string? RecordedBy
    );

    public record VitalTrendPointDto(DateTime Date, decimal Value, decimal? Value2);
}
