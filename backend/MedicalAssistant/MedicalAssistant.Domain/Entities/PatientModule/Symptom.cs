using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class Symptom : BaseEntity
    {
        public int PatientVisitId { get; set; }
        public PatientVisit? PatientVisit { get; set; }

        public string Name { get; set; } = string.Empty;
        public string Severity { get; set; } = string.Empty;
        public string? Duration { get; set; }
        public string? Onset { get; set; }
        public string? Progression { get; set; }
        public string? Location { get; set; }
        public bool IsChronic { get; set; }
        public string? Notes { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
}
