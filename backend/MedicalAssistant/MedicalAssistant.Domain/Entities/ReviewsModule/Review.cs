using MedicalAssistant.Domain.Entities.DoctorsModule;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.ReviewsModule
{
    public class Review : BaseEntity
    {
        public int DoctorId { get; set; }

        public string Author { get; set; } = string.Empty;

        public int Rating { get; set; }

        public string Comment { get; set; } = string.Empty;

        public DateTime CreatedAt { get; set; }

        // Navigation Property
        public Doctor Doctor { get; set; } = null!;
    }
}
