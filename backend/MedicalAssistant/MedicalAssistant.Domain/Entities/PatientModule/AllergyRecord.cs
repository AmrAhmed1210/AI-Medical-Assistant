using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class AllergyRecord : BaseEntity
    {
        public int PatientId { get; set; }
        public Patient? Patient { get; set; }

        public string AllergyType { get; set; } = string.Empty;
        public string AllergenName { get; set; } = string.Empty;
        public string Severity { get; set; } = string.Empty;
        public string? ReactionDescription { get; set; }
        public DateOnly? FirstObservedDate { get; set; }
        public bool IsActive { get; set; } = true;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
}
