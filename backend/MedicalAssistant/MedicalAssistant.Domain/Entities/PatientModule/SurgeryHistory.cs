using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class SurgeryHistory : BaseEntity
    {
        public int PatientId { get; set; }
        public Patient Patient { get; set; } = null!;

        public string SurgeryName { get; set; } = string.Empty;
        public DateOnly SurgeryDate { get; set; }
        public string? HospitalName { get; set; }
        public string? DoctorName { get; set; }
        public string? Complications { get; set; }
        public string? Notes { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
}
