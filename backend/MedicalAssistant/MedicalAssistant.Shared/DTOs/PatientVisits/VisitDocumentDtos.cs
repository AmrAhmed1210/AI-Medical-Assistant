using System;

namespace MedicalAssistant.Shared.DTOs.PatientVisits
{
    public record VisitDocumentDto(
        int Id,
        int PatientVisitId,
        string DocumentType,
        string Title,
        string FileUrl,
        string FileType,
        string? Description,
        DateTime UploadedAt
    );
}
