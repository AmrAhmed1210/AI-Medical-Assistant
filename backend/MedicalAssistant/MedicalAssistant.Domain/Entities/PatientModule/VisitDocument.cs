using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class VisitDocument : BaseEntity
    {
        public int PatientVisitId { get; set; }
        public PatientVisit PatientVisit { get; set; } = null!;

        public string DocumentType { get; set; } = string.Empty;
        public string Title { get; set; } = string.Empty;
        public string FileUrl { get; set; } = string.Empty;
        public string FileType { get; set; } = string.Empty;
        public string? Description { get; set; }

        public DateTime UploadedAt { get; set; } = DateTime.UtcNow;
    }
}
